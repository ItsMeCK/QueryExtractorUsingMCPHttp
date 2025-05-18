# mcp_server.py
# Main Flask application for the MCP Server.
from dotenv import load_dotenv

load_dotenv()
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Import configurations and utilities from other modules
import config  # Uses the mcp_config_py artifact
from db_utils import execute_query, get_table_schema, get_all_table_names, table_exists
# Updated import to use the LangGraph-based function
from llm_integration import generate_sql_from_natural_language_langgraph as generate_sql_from_natural_language
from file_handler import process_uploaded_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH


# --- API Endpoints ---

@app.route('/tools', methods=['GET'])
def list_tools():
    """Lists available tools/endpoints for the Host LLM."""
    tools = [
        {
            "name": "query_database_natural_language",
            "description": "Executes a natural language query against the database. The MCP server will attempt to convert it to SQL and fetch results. It tries to dynamically identify relevant tables using an LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "natural_language_query": {
                        "type": "string",
                        "description": "The user's query in plain English (e.g., 'show me all users with salary over 50000')."
                    },
                    "target_tables_override": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional. A list of specific table names to consider for the query. If provided, dynamic table identification might be skipped or augmented."
                    }
                },
                "required": ["natural_language_query"]
            }
        },
        {
            "name": "upload_and_populate_table_from_file",
            "description": (
                "Uploads a CSV or Excel file. The filename (without extension) is used as the table name. "
                "If the table doesn't exist, it's created based on inferred schema from file headers. "
                "Data from the file is then inserted into the table."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",  # In OpenAPI spec, this would be format: binary for actual file upload
                        "description": "The CSV or Excel file to upload."
                    }
                },
                "required": ["file"]
            }
        },
        {
            "name": "list_database_tables_and_schemas",
            "description": "Retrieves a list of all tables in the database and their detailed schemas (columns, types).",
            "parameters": {}  # No parameters needed
        },
        {
            "name": "check_table_existence",
            "description": "Checks if a specific table exists in the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "The name of the table to check."}
                },
                "required": ["table_name"]
            }
        }
    ]
    return jsonify({"tools": tools, "database_name": config.DB_NAME}), 200


@app.route('/query', methods=['POST'])
def handle_query():
    """
    Receives a natural language query, converts it to SQL (via LLM and LangGraph),
    executes it, and returns the results.
    """
    data = request.get_json()
    if not data or 'natural_language_query' not in data:
        return jsonify({"error": "Missing 'natural_language_query' in request body"}), 400

    nl_query = data['natural_language_query']
    # Optional: User/Host can specify tables to narrow down the LLM's focus
    target_tables_override = data.get('target_tables_override')  # This will be List[str] or None

    # Call the LangGraph-based SQL generation function
    # It's expected to return (sql_query_str | None, error_message_str | None)
    sql_query, error_msg_gen = generate_sql_from_natural_language(nl_query, target_tables_override)

    if error_msg_gen:
        # If LLM/LangGraph pipeline returned an error message
        return jsonify({"error": f"SQL generation process failed: {error_msg_gen}",
                        "details": "Could not translate natural language to SQL."}), 500
    if not sql_query:
        # This case might occur if the LLM pipeline returns (None, None)
        return jsonify({"error": "Failed to generate SQL query (LLM returned no query and no explicit error).",
                        "details": "The LLM was unable to produce a SQL query for the given input."}), 500

    # Proceed to execute the generated SQL query
    results, db_error = execute_query(sql_query, fetch_all=True)

    if db_error:
        return jsonify({"error": f"Database error: {db_error}", "generated_sql": sql_query}), 500

    # `results` can be an empty list if query is valid but returns no rows, which is not an error.
    # `results` can be None if `execute_query` itself had an issue not caught as db_error.
    if results is None and not db_error:
        return jsonify(
            {"error": "Query executed but no results returned and no explicit database error (check server logs).",
             "generated_sql": sql_query}), 500

    return jsonify({"generated_sql": sql_query, "results": results if results is not None else []}), 200


@app.route('/upload_data', methods=['POST'])
def upload_data_endpoint():
    """
    Handles file upload (CSV/Excel) using the file_handler module.
    Filename (without extension) becomes the table name.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file_storage = request.files['file']
    if file_storage.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Extract filename without extension for table name
    # secure_filename helps prevent directory traversal attacks and normalizes the filename.
    filename_sans_ext = os.path.splitext(secure_filename(file_storage.filename))[0]
    if not filename_sans_ext:  # Handle cases like ".csv" or just "."
        return jsonify({"error": "Invalid filename for deriving table name."}), 400

    response_data, status_code = process_uploaded_file(file_storage, filename_sans_ext)
    return jsonify(response_data), status_code


@app.route('/tables', methods=['GET'])
def get_all_tables_endpoint():
    """Gets all tables and their schemas from the database."""
    all_table_names, error = get_all_table_names()
    if error:
        return jsonify({"error": f"Failed to retrieve table names: {error}"}), 500

    tables_data_with_schemas = {}
    if not all_table_names:  # No tables found
        return jsonify(
            {"message": "No tables found in the database.", "tables": {}, "database_name": config.DB_NAME}), 200

    for table_name in all_table_names:
        schema, schema_error = get_table_schema(table_name)
        if schema_error:
            tables_data_with_schemas[table_name] = {"error": schema_error, "schema": []}
        else:
            # Ensure schema is always a list, even if empty (e.g., table exists but has no columns, though unlikely)
            tables_data_with_schemas[table_name] = {"schema": schema if schema is not None else []}

    return jsonify({"tables": tables_data_with_schemas, "database_name": config.DB_NAME}), 200


@app.route('/table/<path:table_name>/exists', methods=['GET'])
def check_table_exists_endpoint(table_name):
    """
    Checks if a specific table exists.
    Using <path:table_name> to allow for table names that might contain characters
    that default string converter might not handle well, though secure_filename will sanitize.
    """
    # Sanitize the table_name to prevent any malicious inputs, although table_exists should use parameterized queries or safe methods.
    sane_table_name = secure_filename(table_name)
    if not sane_table_name or sane_table_name != table_name.replace('/',
                                                                    '_'):  # Check if secure_filename significantly altered it in an unexpected way for simple names
        # This check is basic. Complex names might be altered by secure_filename.
        # The main goal is to prevent obviously malicious inputs.
        # If table names can have special chars that secure_filename removes, this logic might need adjustment
        # or rely purely on how `table_exists` handles the name.
        print(f"Original table name '{table_name}' sanitized to '{sane_table_name}'")
        # If sane_table_name is empty after sanitization, it's invalid.
        if not sane_table_name:
            return jsonify({"error": "Invalid table name provided (empty after sanitization)."}), 400

    exists = table_exists(sane_table_name)  # db_utils.table_exists should handle SQL safety
    return jsonify({"table_name": sane_table_name, "requested_name": table_name, "exists": exists,
                    "database_name": config.DB_NAME}), 200


# --- Main ---
if __name__ == '__main__':
    # Ensure OPENAI_API_KEY is set in the environment for llm_integration to work
    if not os.getenv('OPENAI_API_KEY'):
        print("WARNING: OPENAI_API_KEY environment variable is not set. LLM features will likely fail.")

    print(f"Starting MCP Server on {config.SERVER_HOST}:{config.SERVER_PORT}")
    print(f"Debug mode: {config.DEBUG_MODE}")
    print(
        f"Attempting to connect to DB: mysql://{config.DB_USER}:****@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}")

    # Optional: Test DB connection on startup (can be moved to a health check endpoint)
    # from db_utils import get_db_connection
    # conn_test = get_db_connection()
    # if conn_test:
    #     print("Successfully connected to the database on startup.")
    #     conn_test.close()
    # else:
    #     print("CRITICAL: Failed to connect to the database on startup. Please check configurations and ensure MySQL server is running.")
    # Consider exiting if DB is essential: exit(1)

    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=config.DEBUG_MODE)
