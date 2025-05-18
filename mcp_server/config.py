# config.py
# Configuration settings for the MCP Server

import os

# --- Database Configuration ---
# Load database configuration from environment variables for security
# Defaults are provided for local development if environment variables are not set.
DB_HOST = os.environ.get('MYSQL_HOST', '127.0.0.1')
DB_PORT = int(os.environ.get('MYSQL_PORT', 3306))
DB_USER = os.environ.get('MYSQL_USER', 'mcp_user')
DB_PASSWORD = os.environ.get('MYSQL_PASSWORD', 'mcp_password')
DB_NAME = os.environ.get('MYSQL_DATABASE', 'user_data_db')

# --- File Upload Settings ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size

# --- LLM Configuration (Example) ---
# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') # Store your API key securely

# --- Server Configuration ---
SERVER_HOST = '0.0.0.0'
SERVER_PORT = int(os.environ.get('MCP_PORT', 5001))
DEBUG_MODE = os.environ.get('FLASK_DEBUG', '1') == '1'

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    try:
        os.makedirs(UPLOAD_FOLDER)
        print(f"Created upload folder: {UPLOAD_FOLDER}")
    except OSError as e:
        print(f"Error creating upload folder {UPLOAD_FOLDER}: {e}")
        # Depending on the severity, you might want to exit or raise an exception
