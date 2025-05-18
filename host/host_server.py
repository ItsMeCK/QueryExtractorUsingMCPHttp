# host_server.py
# This Flask server provides the chat UI and acts as an intelligent backend
# to communicate with the MCP Server using LangGraph for LLM-based tool calling.
from dotenv import load_dotenv

load_dotenv()
import os
import requests  # To make HTTP requests to the MCP server
import json
from typing import List, Dict, TypedDict, Annotated, Sequence
import operator

from flask import Flask, render_template, request, jsonify
# from openai import OpenAI as OpenAIClient  # Renamed to avoid conflict if Langchain's OpenAI is used differently
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from langgraph.graph import StateGraph, END

# from langgraph.checkpoint.memory import MemorySaver # Removed this unused import

app = Flask(__name__)

# --- Configuration ---
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:5000")

# Initialize OpenAI client for direct calls if needed (though Langchain will use its own)
# and Langchain's ChatOpenAI model
try:
    # direct_openai_client = OpenAIClient() # If needed for other non-Langchain calls
    lc_openai_llm = ChatOpenAI(model="gpt-4o", temperature=0)  # Using Langchain's ChatOpenAI
    print("Langchain ChatOpenAI model initialized.")
except Exception as e:
    print(f"Failed to initialize Langchain ChatOpenAI model: {e}. Ensure OPENAI_API_KEY is set.")
    lc_openai_llm = None


# --- Helper Functions (MCP Interaction) ---

def get_mcp_tools_from_server():
    """Fetches the list of available tools from the MCP server."""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/tools", timeout=10)
        response.raise_for_status()
        tools_data = response.json()
        if "tools" in tools_data and isinstance(tools_data["tools"], list):
            return tools_data["tools"]
        else:
            print(f"MCP /tools endpoint returned unexpected data format: {tools_data}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching tools from MCP server: {e}")
        return []


def call_mcp_endpoint_func(tool_name: str, tool_args: dict):
    """Calls the appropriate MCP server endpoint based on the tool name and arguments."""
    print(f"Host: Executing MCP tool '{tool_name}' with args: {tool_args}")
    mcp_response_data = None
    endpoint = ""
    method = "POST"  # Default method

    if tool_name == "query_database_natural_language":
        endpoint = "/query"
        payload = {"natural_language_query": tool_args.get("natural_language_query")}
        if tool_args.get("target_tables_override"):
            payload["target_tables_override"] = tool_args.get("target_tables_override")
    elif tool_name == "list_database_tables_and_schemas":
        endpoint = "/tables"
        method = "GET"
        payload = None  # No payload for this GET request
    elif tool_name == "check_table_existence":
        table_name_arg = tool_args.get("table_name")
        if table_name_arg:
            endpoint = f"/table/{table_name_arg}/exists"  # Path parameter
            method = "GET"
            payload = None  # No payload for this GET request
        else:
            return {"error": "LLM did not provide table_name for check_table_existence."}
    elif tool_name == "upload_and_populate_table_from_file":
        # This remains a placeholder as file upload via pure JSON tool call is complex
        return {"message": "File upload intent recognized. This feature requires a UI file picker."}
    else:
        return {"error": f"Unknown MCP tool name: {tool_name}"}

    # Actual call to MCP
    full_url = f"{MCP_SERVER_URL}{endpoint}"
    try:
        if method.upper() == "POST":
            response = requests.post(full_url, json=payload, timeout=60)
        elif method.upper() == "GET":
            response = requests.get(full_url, params=payload, timeout=30)
        else:  # Should not happen given the logic above
            return {"error": f"Internal: Unsupported HTTP method '{method}' for tool '{tool_name}'."}

        response.raise_for_status()
        mcp_response_data = response.json()
    except requests.exceptions.HTTPError as http_err:
        error_details = f"MCP Server returned an error for {method} {endpoint}."
        try:
            error_details += f" Details: {http_err.response.json().get('error', http_err.response.text)}"
        except:
            pass
        print(f"HTTP error calling MCP {endpoint}: {error_details}")
        return {"error": error_details, "mcp_status_code": http_err.response.status_code}
    except requests.exceptions.RequestException as req_err:
        print(f"Request exception calling MCP {endpoint}: {req_err}")
        return {"error": f"Could not connect to MCP server at {endpoint}."}

    return mcp_response_data


# --- LangGraph State and Nodes ---

class HostGraphState(TypedDict):
    user_message: str
    mcp_tools_definition: List[Dict]  # Tools available from MCP
    invoked_tool_name: str | None  # Name of the tool LLM decided to call
    invoked_tool_args: Dict | None  # Arguments for that tool
    mcp_tool_response: Dict | None  # Raw response from MCP after tool execution
    final_bot_reply: str | None  # User-facing reply
    error_message: str | None  # For errors during graph execution
    # For multi-turn, history would be added:
    # messages: Annotated[Sequence[BaseMessage], operator.add]


# Node: Fetch available tools from MCP
def fetch_mcp_tools_node(state: HostGraphState):
    print("--- HostGraph: Fetching MCP Tools ---")
    tools = get_mcp_tools_from_server()
    if not tools:
        return {"error_message": "Failed to fetch tools from MCP server.", "mcp_tools_definition": []}
    return {"mcp_tools_definition": tools, "error_message": None}


# Node: LLM decides which tool to call (or if to reply directly)
def decide_tool_or_reply_node(state: HostGraphState):
    print("--- HostGraph: LLM Deciding Tool/Reply ---")
    if not lc_openai_llm:
        return {"error_message": "Host LLM (Langchain ChatOpenAI) not initialized."}
    if not state.get("mcp_tools_definition"):
        return {"error_message": "No MCP tools available for LLM to choose from."}

    system_prompt = "You are a helpful assistant. Based on the user's query and the available specialized tools, decide if a tool should be used. If a tool is appropriate, specify which one and its arguments. If no tool is suitable, or if the query is a greeting or general chit-chat, you can reply directly. Only choose one tool if multiple seem applicable."

    llm_with_tools = lc_openai_llm.bind_tools(tools=state["mcp_tools_definition"])

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["user_message"])
    ]

    try:
        ai_response_message = llm_with_tools.invoke(messages)

        if ai_response_message.tool_calls:
            tool_call = ai_response_message.tool_calls[0]
            print(f"LLM chose tool: {tool_call['name']} with args: {tool_call['args']}")
            return {
                "invoked_tool_name": tool_call["name"],
                "invoked_tool_args": tool_call["args"],
                "final_bot_reply": None,
                "error_message": None
            }
        else:
            print(f"LLM direct reply: {ai_response_message.content}")
            return {
                "invoked_tool_name": None,
                "invoked_tool_args": None,
                "final_bot_reply": ai_response_message.content,
                "error_message": None
            }
    except Exception as e:
        error_msg = f"Error during LLM tool decision: {str(e)}"
        print(error_msg)
        return {"error_message": error_msg}


# Node: Execute the chosen MCP tool
def execute_mcp_tool_node(state: HostGraphState):
    print("--- HostGraph: Executing MCP Tool ---")
    tool_name = state.get("invoked_tool_name")
    tool_args = state.get("invoked_tool_args")

    if not tool_name:
        return {"error_message": "No tool was chosen for execution."}

    response = call_mcp_endpoint_func(tool_name, tool_args or {})
    return {"mcp_tool_response": response, "error_message": response.get("error")}


# Node: Format the MCP response (or direct LLM reply) for the user
def format_final_reply_node(state: HostGraphState):
    print("--- HostGraph: Formatting Final Reply ---")
    if state.get("error_message") and not state.get("final_bot_reply"):
        return {"final_bot_reply": f"An error occurred: {state['error_message']}"}

    if state.get("final_bot_reply"):
        return {}

    mcp_response = state.get("mcp_tool_response")
    tool_name = state.get("invoked_tool_name")
    bot_reply_content = ""

    if not mcp_response:
        bot_reply_content = "No response received from the data processor after tool execution."
    elif "error" in mcp_response:
        bot_reply_content = f"Error from data processor: {mcp_response['error']}"
        if mcp_response.get("mcp_status_code"):
            bot_reply_content += f" (Status: {mcp_response['mcp_status_code']})"
        if "generated_sql" in mcp_response:
            bot_reply_content += f"\nAttempted SQL: `{mcp_response['generated_sql']}`"
    elif tool_name == "query_database_natural_language":
        if "results" in mcp_response and mcp_response["results"]:
            formatted_results = []
            if isinstance(mcp_response["results"], list):
                for row in mcp_response["results"]:
                    if isinstance(row, dict):
                        formatted_results.append(", ".join([f"{k}: {v}" for k, v in row.items()]))
                    else:
                        formatted_results.append(str(row))
                bot_reply_content = f"Generated SQL: `{mcp_response.get('generated_sql', 'N/A')}`\n\nResults:\n" + "\n".join(
                    formatted_results)
            else:
                bot_reply_content = f"Generated SQL: `{mcp_response.get('generated_sql', 'N/A')}`\n\nResponse: {mcp_response.get('results', 'No results field')}"
        elif "generated_sql" in mcp_response:
            bot_reply_content = f"Generated SQL: `{mcp_response.get('generated_sql')}`\n\nQuery executed. No data returned or action performed."
        else:
            bot_reply_content = f"Processed query. Response: {str(mcp_response)}"
    elif tool_name == "list_database_tables_and_schemas":
        if "tables" in mcp_response and mcp_response["tables"]:
            tables_list = [
                f"- **{name}**: Columns: {', '.join([c.get('Field', 'N/A') for c in s.get('schema', [])]) if s.get('schema') else 'No column info'}"
                for name, s in mcp_response["tables"].items()]
            bot_reply_content = "Available tables and their primary columns:\n" + "\n".join(tables_list)
            if mcp_response.get(
                "database_name"): bot_reply_content += f"\n\n(Database: {mcp_response['database_name']})"
        else:
            bot_reply_content = "Could not retrieve table list or no tables found."
    elif tool_name == "check_table_existence":
        if "exists" in mcp_response:
            status = "exists" if mcp_response["exists"] else "does not exist"
            bot_reply_content = f"Table '{mcp_response.get('table_name', 'Unknown')}' {status} in database '{mcp_response.get('database_name', 'N/A')}'."
        else:
            bot_reply_content = f"Could not determine existence for table. Response: {mcp_response}"
    elif "message" in mcp_response:
        bot_reply_content = mcp_response["message"]
    else:
        bot_reply_content = f"Received an unformatted response from data processor for tool {tool_name}: {str(mcp_response)}"

    return {"final_bot_reply": bot_reply_content}


# --- LangGraph Conditional Edges ---
def should_execute_tool(state: HostGraphState):
    print("--- HostGraph Condition: Should Execute Tool? ---")
    if state.get("error_message"): return END
    if state.get("invoked_tool_name"):
        print(f"Yes, tool '{state['invoked_tool_name']}' was chosen.")
        return "execute_mcp_tool"
    print("No, LLM decided to reply directly or no tool chosen.")
    return "format_final_reply"


# --- Build LangGraph Workflow ---
if lc_openai_llm:
    host_workflow = StateGraph(HostGraphState)
    host_workflow.add_node("fetch_mcp_tools", fetch_mcp_tools_node)
    host_workflow.add_node("decide_tool_or_reply", decide_tool_or_reply_node)
    host_workflow.add_node("execute_mcp_tool", execute_mcp_tool_node)
    host_workflow.add_node("format_final_reply", format_final_reply_node)

    host_workflow.set_entry_point("fetch_mcp_tools")
    host_workflow.add_edge("fetch_mcp_tools", "decide_tool_or_reply")
    host_workflow.add_conditional_edges(
        "decide_tool_or_reply",
        should_execute_tool,
        {
            "execute_mcp_tool": "execute_mcp_tool",
            "format_final_reply": "format_final_reply"
        }
    )
    host_workflow.add_edge("execute_mcp_tool", "format_final_reply")
    host_workflow.add_edge("format_final_reply", END)

    try:
        host_app_graph = host_workflow.compile()
        print("Host LangGraph app compiled successfully.")
    except Exception as e:
        print(f"Error compiling Host LangGraph: {e}")
        host_app_graph = None
else:
    host_app_graph = None


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def handle_chat_message():
    if not host_app_graph:
        return jsonify({"reply": "Host server's LangGraph application is not initialized."}), 500

    data = request.get_json()
    user_message = data.get('message')
    if not user_message:
        return jsonify({"error": "No message received"}), 400

    print(f"Host Server (LangGraph) received message: {user_message}")

    initial_state = HostGraphState(
        user_message=user_message,
        mcp_tools_definition=[],
        invoked_tool_name=None,
        invoked_tool_args=None,
        mcp_tool_response=None,
        final_bot_reply=None,
        error_message=None
    )

    try:
        final_state = host_app_graph.invoke(initial_state, {"recursion_limit": 10})

        if final_state.get("error_message") and not final_state.get("final_bot_reply"):
            reply_content = f"An error occurred: {final_state['error_message']}"
        elif final_state.get("final_bot_reply"):
            reply_content = final_state["final_bot_reply"]
        else:
            reply_content = "I'm sorry, I encountered an issue and could not process your request."

        return jsonify({"reply": reply_content})

    except Exception as e:
        print(f"An unexpected error occurred in Host server /chat (LangGraph): {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"reply": f"An internal error occurred in the Host server: {str(e)}"}), 500


if __name__ == '__main__':
    if not os.path.exists("templates"):
        os.makedirs("templates")
        print("INFO: 'templates' directory created. Ensure 'index.html' is inside it.")

    if not lc_openai_llm:
        print(
            "CRITICAL: Langchain ChatOpenAI model failed to initialize. Host server's LLM capabilities will not work.")
        print("Please ensure your OPENAI_API_KEY environment variable is correctly set.")

    app.run(debug=True, host='0.0.0.0', port=5001)
