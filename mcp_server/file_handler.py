# file_handler.py
# Handles file uploads, parsing, and table creation/population logic.
from dotenv import load_dotenv

load_dotenv()
import os
import io
import pandas as pd
from werkzeug.utils import secure_filename
from config import ALLOWED_EXTENSIONS, UPLOAD_FOLDER
from db_utils import table_exists, execute_query, get_db_connection # Added get_db_connection

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def infer_sql_schema_from_dataframe(df, table_name):
    """
    Infers a basic SQL CREATE TABLE statement from a Pandas DataFrame.
    Maps Pandas dtypes to MySQL types.
    """
    type_mapping = {
        'int64': 'INT',
        'int32': 'INT',
        'int16': 'INT',
        'int8': 'INT',
        'float64': 'DOUBLE', # Using DOUBLE for more precision than FLOAT
        'float32': 'FLOAT',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'DATETIME',
        'timedelta[ns]': 'TEXT', # Or convert to seconds (INT) or string representation
        'object': 'TEXT',  # Default for strings or mixed types
        'category': 'TEXT' # Or VARCHAR(255) if max length is known
    }
    columns_sql = []
    for column_name, dtype in df.dtypes.items():
        # Sanitize column name (basic): replace non-alphanumeric with underscore
        sane_column_name = "".join(c if c.isalnum() or c == '_' else '_' for c in str(column_name))
        if not sane_column_name: # Handle empty or fully invalid column names
            sane_column_name = f"column_{len(columns_sql)}"

        sql_type = type_mapping.get(str(dtype).lower(), 'TEXT')
        columns_sql.append(f"`{sane_column_name}` {sql_type}")

    if not columns_sql:
        raise ValueError("DataFrame has no columns to create a table.")

    # Table name is already sanitized in the calling function (mcp_server.py)
    create_table_sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({', '.join(columns_sql)})"
    return create_table_sql, [col.split('`')[1] for col in columns_sql] # return sanitized column names

def process_uploaded_file(file_storage, filename_sans_ext):
    """
    Processes an uploaded file (CSV/Excel).
    - Reads data into a Pandas DataFrame.
    - Infers schema and creates table if not exists.
    - Inserts data into the table.
    Returns:
        tuple: (message_dict, status_code)
    """
    if not file_storage or not file_storage.filename:
        return {"error": "No file provided"}, 400

    original_filename = file_storage.filename
    if not allowed_file(original_filename):
        return {"error": "File type not allowed"}, 400

    # Sanitize table name from filename_sans_ext (already done in endpoint, but good for robustness)
    table_name = "".join(c if c.isalnum() or c == '_' else '_' for c in filename_sans_ext)
    if not table_name: # if filename was purely special characters
        return {"error": "Invalid table name derived from filename."}, 400

    try:
        file_content = file_storage.read() # Read file content into memory
        if original_filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content))
        elif original_filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_content))
        else:
            # This case should ideally be caught by allowed_file, but as a safeguard:
            return {"error": "Unsupported file type (internal check failed)"}, 400
    except Exception as e:
        return {"error": f"Error reading file content: {str(e)}"}, 400

    if df.empty:
        return {"message": f"File '{original_filename}' is empty. Table '{table_name}' was not modified or created if it didn't exist."}, 200

    # Clean DataFrame column names for SQL compatibility
    original_columns = df.columns.tolist()
    df.columns = ["".join(c if c.isalnum() or c == '_' else '_' for c in str(col)) for col in df.columns]
    # Handle potentially duplicate column names after sanitization (e.g., "Col A" and "Col_A" both become "Col_A")
    # This simplistic approach appends suffixes. A more robust way might involve user feedback.
    cols = []
    for col in df.columns:
        new_col = col
        counter = 1
        while new_col in cols:
            new_col = f"{col}_{counter}"
            counter += 1
        cols.append(new_col)
    df.columns = cols
    sanitized_column_names = df.columns.tolist()


    conn = get_db_connection()
    if not conn:
        return {"error": "Database connection failed"}, 500
    cursor = conn.cursor() # Not using dictionary cursor for DDL/DML

    try:
        status_msg = ""
        created_new_table = False
        if not table_exists(table_name):
            create_table_sql, schema_sanitized_cols = infer_sql_schema_from_dataframe(df, table_name)
            # The schema_sanitized_cols from infer_sql_schema_from_dataframe should match df.columns
            # if the sanitization logic is consistent.
            print(f"Creating table '{table_name}' with SQL: {create_table_sql}")
            _, error = execute_query(create_table_sql, is_ddl=True) # Uses its own connection
            if error:
                raise Exception(f"Failed to create table '{table_name}': {error}")
            status_msg = f"Table '{table_name}' created. "
            created_new_table = True
        else:
            status_msg = f"Table '{table_name}' already exists. "
            # TODO: Optionally, validate if df schema matches existing table schema before inserting.
            # For now, we assume it matches or user wants to append compatible data.

        # Insert data
        # Ensure column names used in INSERT match the DataFrame's sanitized column names
        cols_for_insert = ', '.join([f"`{col}`" for col in sanitized_column_names])
        placeholders = ', '.join(['%s'] * len(sanitized_column_names))
        insert_sql = f"INSERT INTO `{table_name}` ({cols_for_insert}) VALUES ({placeholders})"

        # Prepare data for insertion, converting Pandas NA types to None (SQL NULL)
        data_tuples = []
        for row_tuple in df.itertuples(index=False, name=None):
            processed_row = [None if pd.isna(val) else val for val in row_tuple]
            data_tuples.append(tuple(processed_row))

        if data_tuples:
            cursor.executemany(insert_sql, data_tuples)
            conn.commit()
            status_msg += f"{len(data_tuples)} rows inserted into '{table_name}'."
            return {
                "message": status_msg,
                "table_name": table_name,
                "rows_inserted": len(data_tuples),
                "created_new_table": created_new_table,
                "original_columns": original_columns,
                "sanitized_columns": sanitized_column_names
            }, 201
        else:
            status_msg += "No data to insert."
            return {"message": status_msg, "table_name": table_name, "rows_inserted": 0}, 200

    except pd.errors.EmptyDataError:
        return {"message": f"File '{original_filename}' is empty or has no data rows. No action taken for table '{table_name}'."}, 200
    except Exception as e:
        if conn and conn.is_connected(): # Check if conn is valid before rollback
            try:
                conn.rollback()
            except Error as rb_err:
                print(f"Error during rollback: {rb_err}")
        return {"error": f"An error occurred during file processing for table '{table_name}': {str(e)}"}, 500
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

