# llm_integration.py
# Handles interactions with LLMs for SQL generation and other NLP tasks using Langchain and LangGraph.

import os
import datetime
from typing import List, Dict, TypedDict, Sequence
import operator  # Added for potential use in LangGraph, though not strictly needed by current example
import re  # Import regular expressions module

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from langgraph.graph import StateGraph, END

# Assuming db_utils.py and config.py are in the same directory
from db_utils import get_table_schema, get_all_table_names
from config import DB_NAME  # OPENAI_API_KEY is loaded by ChatOpenAI if set in env

# --- Initialize LLM ---
try:
    llm = ChatOpenAI(temperature=0, model="gpt-4o")  # Or "gpt-3.5-turbo"
    print("ChatOpenAI model initialized successfully.")
except Exception as e:
    print(f"Error initializing ChatOpenAI: {e}. Please ensure OPENAI_API_KEY is set and valid.")
    llm = None


# --- Pydantic Models for Structured Output ---
class RelevantTables(BaseModel):
    """Schema for extracting relevant table names from a query."""
    table_names: List[str] = Field(
        description="A list of table names deemed relevant to the user's query, selected from the provided list of available tables.")


class SQLQuery(BaseModel):
    """Schema for the generated SQL query."""
    sql_query: str = Field(description="The generated SQL query.")
    error: str = Field(description="Error message if SQL generation failed, otherwise empty string.", default="")


# --- Langchain Functions ---

def lc_identify_relevant_tables(natural_language_query: str, all_table_names: List[str]) -> RelevantTables:
    """
    Uses Langchain and an LLM to identify relevant table names from a natural language query,
    constrained by a list of available tables.
    """
    if not llm:
        raise ValueError("LLM not initialized. Cannot identify relevant tables.")
    if not all_table_names:
        return RelevantTables(table_names=[])

    parser = JsonOutputParser(pydantic_object=RelevantTables)
    format_instructions = parser.get_format_instructions()
    escaped_format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert at understanding database schemas and user queries. "
         "Your task is to identify which of the available tables are relevant to the user's question. "
         "Only choose from the provided list of table names. "
         "If no tables seem relevant, return an empty list for 'table_names'.\n"
         f"{escaped_format_instructions}"),
        ("human",
         "Available table names: {all_table_names}\n\n"
         "User query: \"{natural_language_query}\"\n\n"
         "Based on the query and the available tables, which tables are relevant? (Respond in the requested JSON format)")
    ])
    chain = prompt_template | llm | parser
    try:
        response = chain.invoke({
            "all_table_names": ", ".join(all_table_names),
            "natural_language_query": natural_language_query
        })
        return RelevantTables(table_names=response.get("table_names", []))
    except Exception as e:
        print(f"Error during Langchain table identification: {e}")
        raise e


def lc_generate_sql(natural_language_query: str, table_schemas: Dict[str, List[Dict]],
                    db_dialect: str = "MySQL") -> SQLQuery:
    """
    Uses Langchain and an LLM to generate SQL from a natural language query,
    given specific table schemas.
    """
    if not llm:
        raise ValueError("LLM not initialized. Cannot generate SQL.")
    if not table_schemas:
        print("Warning: Generating SQL without specific table schemas. Quality may be affected.")

    current_year = datetime.datetime.now().year
    previous_year = current_year - 1
    output_parser = StrOutputParser()
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         f"You are an expert {db_dialect} SQL query generator. Your task is to translate the user's natural language query into an executable {db_dialect} SQL query, using ONLY the provided table schemas. "
         "If the query involves dates like 'previous year', assume current year is {current_year} and previous year is {previous_year}. "
         "Ensure monetary values are treated as numbers. "
         "Do not use any tables or columns not explicitly mentioned in the provided schemas. "
         "If the provided schemas are insufficient to answer the query, explain why instead of generating a potentially incorrect SQL query. "
         "Output ONLY the SQL query, without any additional explanation, comments, or markdown formatting. "  # Reinforce no markdown
         "If you cannot generate a valid query based on the provided information, return a short error message starting with 'ERROR:'."),
        ("human",
         "Database Schemas:\n"
         "{table_schemas_str}\n\n"
         "User Query: \"{natural_language_query}\"\n\n"
         "Generated {db_dialect} SQL Query:")
    ])
    chain = prompt_template | llm | output_parser
    schemas_str_parts = []
    if table_schemas:
        for table_name, columns in table_schemas.items():
            col_defs = []
            for col in columns:
                col_defs.append(f"  {col.get('Field', 'UnknownColumn')} ({col.get('Type', 'UnknownType')})")
            schemas_str_parts.append(f"Table `{table_name}`:\n" + "\n".join(col_defs))
        table_schemas_str = "\n\n".join(schemas_str_parts)
    else:
        table_schemas_str = "No specific table schemas provided. Attempt to generate a general query if possible, or state if schemas are needed."
    try:
        generated_sql_raw = chain.invoke({  # Renamed to generated_sql_raw
            "table_schemas_str": table_schemas_str,
            "natural_language_query": natural_language_query,
            "current_year": current_year,
            "previous_year": previous_year,
            "db_dialect": db_dialect
        })

        # ** NEW: Strip Markdown code fences and leading/trailing whitespace **
        # Regex to find content within ```sql ... ``` or ``` ... ```
        match = re.search(r"```(?:sql\s*)?(.*?)```", generated_sql_raw, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned_sql = match.group(1).strip()
        else:
            cleaned_sql = generated_sql_raw.strip()  # If no fences, just strip whitespace

        if cleaned_sql.strip().upper().startswith("ERROR:"):
            return SQLQuery(sql_query="", error=cleaned_sql.strip())
        return SQLQuery(sql_query=cleaned_sql)  # Return the cleaned SQL
    except Exception as e:
        print(f"Error during Langchain SQL generation: {e}")
        raise e


# --- LangGraph State and Nodes ---

class GraphState(TypedDict):
    natural_language_query: str
    target_tables_override: List[str] | None
    all_table_names_in_db: List[str]
    relevant_table_names: List[str]
    table_schemas: Dict[str, List[Dict]]
    generated_sql: str | None
    error_message: str | None


def get_all_tables_node(state: GraphState):
    print("--- Running Node: Get All Tables ---")
    all_names, err = get_all_table_names()
    if err:
        return {"error_message": f"Failed to retrieve table list: {err}", "all_table_names_in_db": []}
    return {"all_table_names_in_db": all_names, "error_message": None}


def identify_tables_node(state: GraphState):
    print("--- Running Node: Identify Tables ---")
    try:
        if state.get("target_tables_override"):
            print(f"Using overridden target tables: {state['target_tables_override']}")
            valid_overrides = [name for name in state['target_tables_override'] if
                               name in state['all_table_names_in_db']]
            if len(valid_overrides) != len(state['target_tables_override']):
                print("Warning: Some overridden target tables do not exist in the database.")
            return {"relevant_table_names": valid_overrides, "error_message": None}
        if not state.get("all_table_names_in_db"):
            print("No tables in DB to identify from.")
            return {"relevant_table_names": [], "error_message": "No tables available in the database to query."}
        relevant_tables_result = lc_identify_relevant_tables(
            state["natural_language_query"],
            state["all_table_names_in_db"]
        )
        return {"relevant_table_names": relevant_tables_result.table_names, "error_message": None}
    except Exception as e:
        error_msg = f"Error in identify_tables_node: {str(e)}"
        print(error_msg)
        return {"relevant_table_names": [], "error_message": error_msg}


def fetch_schemas_node(state: GraphState):
    print("--- Running Node: Fetch Schemas ---")
    if not state.get("relevant_table_names"):
        print("No relevant tables identified to fetch schemas for. Skipping schema fetching.")
        return {"table_schemas": {}, "error_message": state.get("error_message")}

    schemas = {}
    fetch_errors = []
    for table_name in state["relevant_table_names"]:
        schema, err = get_table_schema(table_name)
        if err:
            fetch_errors.append(f"Could not fetch schema for '{table_name}': {err}")
        elif schema:
            schemas[table_name] = schema
        else:
            schemas[table_name] = []
    current_error = state.get("error_message")
    if fetch_errors:
        print(f"Warning: Errors during schema fetching: {'; '.join(fetch_errors)}")
    return {"table_schemas": schemas, "error_message": current_error}


def generate_sql_node(state: GraphState):
    print("--- Running Node: Generate SQL ---")
    if state.get("error_message") and not state.get("relevant_table_names") and not state.get("target_tables_override"):
        print(
            f"Skipping SQL generation due to critical prior error and no tables to work with: {state['error_message']}")
        return {"generated_sql": None}

    try:
        sql_result = lc_generate_sql(
            state["natural_language_query"],
            state["table_schemas"]
        )
        if sql_result.error:
            return {"generated_sql": None, "error_message": sql_result.error}
        return {"generated_sql": sql_result.sql_query, "error_message": None}
    except Exception as e:
        error_msg = f"Error in generate_sql_node: {str(e)}"
        print(error_msg)
        return {"generated_sql": None, "error_message": error_msg}


# --- Conditional Edge Functions ---
def decide_after_identify_tables(state: GraphState):
    """Decision logic after the 'identify_tables_node'."""
    print("--- Conditional Edge: After Identify Tables ---")
    if state.get("error_message"):
        print(f"Error detected: '{state['error_message']}'. Ending graph.")
        return END
    if not state.get("relevant_table_names"):
        print("No relevant tables identified. Ending graph.")
        return END
    print("Relevant tables identified. Proceeding to fetch schemas.")
    return "fetch_schemas"


def decide_after_fetch_schemas(state: GraphState):
    """Decision logic after the 'fetch_schemas_node'."""
    print("--- Conditional Edge: After Fetch Schemas ---")
    if state.get("error_message"):
        print(f"Error detected: '{state['error_message']}'. Ending graph.")
        return END
    print("Schemas processed (or attempted). Proceeding to generate SQL.")
    return "generate_sql"


if llm:
    workflow = StateGraph(GraphState)

    workflow.add_node("get_all_tables", get_all_tables_node)
    workflow.add_node("identify_tables", identify_tables_node)
    workflow.add_node("fetch_schemas", fetch_schemas_node)
    workflow.add_node("generate_sql", generate_sql_node)

    workflow.set_entry_point("get_all_tables")
    workflow.add_edge("get_all_tables", "identify_tables")

    workflow.add_conditional_edges(
        "identify_tables",
        decide_after_identify_tables,
        {
            "fetch_schemas": "fetch_schemas",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "fetch_schemas",
        decide_after_fetch_schemas,
        {
            "generate_sql": "generate_sql",
            END: END
        }
    )
    workflow.add_edge("generate_sql", END)

    try:
        app_graph = workflow.compile()
        print("LangGraph app compiled successfully.")
    except Exception as e:
        print(f"Error compiling LangGraph: {e}")
        app_graph = None
else:
    app_graph = None
    print("LLM not initialized, LangGraph app not compiled.")


def generate_sql_from_natural_language_langgraph(natural_language_query: str,
                                                 target_tables_override: List[str] | None = None):
    if not app_graph:
        return None, "LLM or LangGraph app not initialized. Cannot generate SQL."

    initial_state = GraphState(
        natural_language_query=natural_language_query,
        target_tables_override=target_tables_override,
        all_table_names_in_db=[],
        relevant_table_names=[],
        table_schemas={},
        generated_sql=None,
        error_message=None
    )
    try:
        final_state = app_graph.invoke(initial_state, {"recursion_limit": 10})
        if final_state.get("error_message"):
            print(f"LangGraph execution resulted in error: {final_state['error_message']}")
            return None, final_state["error_message"]
        if final_state.get("generated_sql"):
            return final_state["generated_sql"], None
        else:
            return None, final_state.get("error_message",
                                         "SQL generation did not complete successfully (no SQL and no explicit error).")
    except Exception as e:
        print(f"Exception during LangGraph execution: {e}")
        return None, f"LangGraph invocation failed: {str(e)}"
