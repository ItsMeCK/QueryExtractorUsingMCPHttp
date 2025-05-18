# db_utils.py
# Database interaction utilities for the MCP Server
from dotenv import load_dotenv

load_dotenv()
import mysql.connector
from mysql.connector import Error
from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

def get_db_connection():
    """Establishes a connection to the MySQL database."""
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        if conn.is_connected():
            # print("Successfully connected to the database.")
            return conn
    except Error as e:
        print(f"Error connecting to MySQL database at {DB_HOST}:{DB_PORT} as {DB_USER}: {e}")
        return None

def execute_query(query, params=None, fetch_one=False, fetch_all=False, is_ddl=False):
    """
    Executes a given SQL query.
    Args:
        query (str): The SQL query to execute.
        params (tuple, optional): Parameters for the SQL query to prevent injection.
        fetch_one (bool): True if expecting a single row result.
        fetch_all (bool): True if expecting multiple rows.
        is_ddl (bool): True if the query is a DDL statement (e.g., CREATE TABLE)
                       that doesn't return rows but needs commit.
    Returns:
        Result of the query or None, and an error message string or None.
    """
    conn = get_db_connection()
    if not conn:
        return None, "Database connection failed."

    # Use dictionary=True for fetch_one and fetch_all to get results as dictionaries
    cursor = conn.cursor(dictionary=(fetch_all or fetch_one))
    result = None
    error_message = None

    try:
        # print(f"Executing query: {query} with params: {params}")
        cursor.execute(query, params or ())
        if is_ddl or (not fetch_one and not fetch_all and not query.strip().upper().startswith("SELECT")):
            conn.commit()
            result = {"message": "Query executed successfully."}
            # print("DDL or non-select query committed.")
        elif fetch_one:
            result = cursor.fetchone()
            # print(f"Fetched one: {result}")
        elif fetch_all:
            result = cursor.fetchall()
            # print(f"Fetched all: {len(result) if result else 0} rows.")

    except Error as e:
        print(f"Error executing query '{query[:100]}...': {e}")
        error_message = str(e)
        try:
            conn.rollback() # Rollback in case of error
            # print("Transaction rolled back due to error.")
        except Error as rb_err:
            print(f"Error during rollback: {rb_err}")
            # If rollback fails, the connection might be in an unusable state.
            # It's often best to close it and attempt to re-establish for subsequent operations.
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
            # print("Database connection closed.")
    return result, error_message

def table_exists(table_name):
    """Checks if a table exists in the database."""
    # Sanitize table_name for SHOW TABLES LIKE (basic protection)
    # A more robust way would be to query information_schema.tables
    # query = "SHOW TABLES LIKE %s"
    # result, _ = execute_query(query, (table_name,), fetch_one=True)
    # return result is not None

    # Using information_schema for better reliability and security
    query = """
        SELECT COUNT(*) as count
        FROM information_schema.tables
        WHERE table_schema = %s AND table_name = %s
    """
    # DB_NAME needs to be passed here, or the connection should be to the specific DB
    result, error = execute_query(query, (DB_NAME, table_name), fetch_one=True)
    if error:
        print(f"Error checking if table '{table_name}' exists: {error}")
        return False
    return result and result['count'] > 0


def get_table_schema(table_name):
    """Retrieves the schema of a given table (column names and types)."""
    if not table_exists(table_name): # This check uses DB_NAME internally
        return None, f"Table '{table_name}' does not exist in database '{DB_NAME}'."

    # Using DESCRIBE is generally safe if table_name is confirmed to exist.
    # However, ensure table_name is not user-supplied directly into the f-string without validation.
    # Since table_exists validates it, this should be okay.
    query = f"DESCRIBE `{DB_NAME}`.`{table_name}`;"
    schema, error = execute_query(query, fetch_all=True)

    if error:
        return None, error
    if not schema:
        return [], None # Return empty list if schema is empty but no error

    # Standardize schema output if needed, e.g., to a list of dicts with 'name' and 'type'
    # The `dictionary=True` in execute_query already gives us dicts.
    # Example: [{'Field': 'id', 'Type': 'int(11)', ...}, ...]
    return schema, None

def get_all_table_names():
    """Retrieves a list of all table names in the database."""
    query = "SHOW TABLES"
    tables_tuples, error = execute_query(query, fetch_all=True) # dictionary=True from execute_query
    if error:
        print(f"Error fetching all table names: {error}")
        return [], error
    if not tables_tuples:
        return [], None

    # When dictionary=True is used, each row is a dict.
    # The key for the table name is usually 'Tables_in_{database_name}'
    table_key = f'Tables_in_{DB_NAME}'
    table_names = [row[table_key] for row in tables_tuples if table_key in row]
    return table_names, None
