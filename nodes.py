import re
import json
from decimal import Decimal
from typing import Dict, Any, List
import psycopg2
import datetime
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from config import DB_POOL, llm, buffer_memory, summary_memory, vector_store, State
from utils import schema

# Utility functions
def clean_sql_query(query: str) -> str:
    """Cleans AI-generated SQL query by removing comments and extra whitespace."""
    query = re.sub(r'--.*?\n|/\*.*?\*/', '', query, flags=re.DOTALL)
    return query.replace("```sql", "").replace("```", "").strip().lower()

def update_memory(user_input: str, response: str):
    """Updates buffer memory and summarizes if exceeding 5 interactions."""
    buffer_memory.save_context({"input": user_input}, {"output": response})
    messages = buffer_memory.load_memory_variables({})["history"]
    if len(messages) > 10:  # 5 pairs = 10 messages
        older_messages = messages[:-10]
        recent_messages = messages[-10:]
        older_text = "\n".join([f"Human: {m.content}" if i % 2 == 0 else f"AI: {m.content}" for i, m in enumerate(older_messages)])
        summary_memory.save_context({"input": older_text}, {"output": ""})
        summary = summary_memory.load_memory_variables({})["history"]
        buffer_memory.clear()
        buffer_memory.save_context({"input": f"Summary of earlier conversation: {summary}"}, {"output": ""})
        for i in range(0, len(recent_messages), 2):
            buffer_memory.save_context(
                {"input": recent_messages[i].content},
                {"output": recent_messages[i + 1].content if i + 1 < len(recent_messages) else ""}
            )

def run_sql_query(sql_query: str) -> tuple[str, int]:
    """Executes SQL query safely and returns results as JSON plus rows affected."""
    conn = None
    try:
        conn = DB_POOL.getconn()
        if not conn:
            return json.dumps({"error": "Database connection failed"}), 0

        with conn.cursor() as cursor:
            sql_query = sql_query.strip()
            sql_query = re.sub(
                r"WHERE\s+(\w+)\s*=\s*('[^']+'|\w+)",
                lambda m: f"WHERE LOWER({m.group(1)}) = LOWER({m.group(2)})",
                sql_query,
                flags=re.IGNORECASE
            )
            sql_query = re.sub(r"\bLIKE\b", "ILIKE", sql_query, flags=re.IGNORECASE)
            sql_query = sql_query.lower()

            print("Executing SQL Query:", sql_query)
            cursor.execute(sql_query)

            query_type = sql_query.split()[0].lower()
            if query_type in {"update", "insert", "delete"}:
                rows_affected = cursor.rowcount
                conn.commit()
                return json.dumps({"message": "Query executed successfully", "rows_affected": rows_affected}), rows_affected

            if query_type == "select":
                results = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]

                def convert_value(value):
                    if isinstance(value, datetime.date):
                        return value.strftime("%Y-%m-%d")
                    elif isinstance(value, datetime.time):  # Fixed: Changed query_type to value
                        return value.strftime("%H:%M:%S")
                    elif isinstance(value, Decimal):
                        return float(value)
                    return value

                data = [
                    dict(zip(column_names, [convert_value(val) for val in row]))
                    for row in results
                ]
                return json.dumps(data, indent=2), len(data)

            return json.dumps({"error": "Unsupported query type"}), 0

    except psycopg2.Error as e:
        return json.dumps({"error": f"SQL Error: {str(e)}"}), 0
    finally:
        if conn:
            DB_POOL.putconn(conn)

# Prompt for SQL generation
query_template = PromptTemplate(
    input_variables=["schema", "user_query", "conversation_history"],
    template="""
    Database schema: {schema}
    Conversation history: {conversation_history}
    Based on the schema and conversation history, generate a precise PostgreSQL query to answer the user's request. Use the conversation history to resolve references (e.g., "this mall" refers to the last mentioned mall), but prioritize the database schema to fetch or update data accurately. Use case-insensitive comparisons with ILIKE for pattern matching and LOWER() for equality comparisons.
    If the user wants to update an offer but hasn't provided enough details (e.g., which offer to update or what to change), return a comment like:
    -- Missing details for update. Please ask the user for the offer title or ID and the new values (e.g., discount percentage, start date, end date).
    If the user specifies an offer by name (e.g., 'winter jacket'), assume it refers to an offer title like 'Winter Jacket Sale' if a close match exists in the context or schema. Update both discount_percentage and description if requested.
    If the user requests multiple actions (e.g., update and list), prioritize the update action and generate a query or comment for it, noting the secondary action (e.g., list) will be handled after update details are provided.
    If the user asks to 'list all the malls' or similar, generate a SELECT query to fetch all records from the malls table (e.g., SELECT mall_id, name, location FROM malls). If the schema does not explicitly list a malls table, assume a table named 'malls' with columns mall_id, name, and location exists.
    If the user asks about customers or loyalty points, assume it is related to the mall's loyalty program and generate a query to fetch the relevant data from the appropriate table (e.g., customers table with loyalty_points column).
    Otherwise, return only the SQL query without explanation.
    User: {user_query}
    """
)

query_chain = query_template | llm | RunnablePassthrough()

# Node Implementations
def input_node(state: State) -> State:
    """Input Node: Passes the user query to the workflow."""
    state["user_input"] = state.get("user_input", "")
    return state

def memory_node(state: State) -> State:
    """Memory Node: Retrieves conversation history."""
    history = buffer_memory.buffer_as_str
    state["conversation_history"] = history if history else "No previous conversation"
    return state

def intent_router_node(state: State) -> State:
    """Intent Router Node: Determines intent and multiple sub-intents."""
    user_input = state["user_input"].lower()
    db_keywords = {"select", "update", "insert", "delete", "where", "from", "table", "offer", "mall", "sale", "discount", "event", "list", "details", "show", "customer", "loyalty", "points", "program"}
    state["intent"] = "generic" if not any(keyword in user_input for keyword in db_keywords) else "database"
    state["sub_intents"] = []
    if "update" in user_input:
        state["sub_intents"].append("update")
    if "select" in user_input or "find" in user_input or "show" in user_input or "list" in user_input or "details" in user_input:
        state["sub_intents"].append("select")
    if "insert" in user_input or "add" in user_input:
        state["sub_intents"].append("insert")
    if "delete" in user_input or "remove" in user_input:
        state["sub_intents"].append("delete")
    return state

def tool_selection_node(state: State) -> State:
    """Tool Selection Node: Selects appropriate tool based on intent."""
    if state["intent"] == "database":
        state["tool"] = "sql_query"
    else:
        state["tool"] = "llm_response"
    return state

def tool_invocation_node(state: State) -> State:
    """Tool Invocation Node: Executes the selected tool."""
    if state["tool"] == "sql_query":
        user_input = state["user_input"].lower()

        # Check conversation history for previous update context
        if state.get("update_pending") and "update" not in state["sub_intents"]:
            history = state["conversation_history"]
            if "update an offer" in history.lower():
                user_input = f"update an offer with {user_input}"

        # Simplified: Always generate a fresh query, bypassing caching for now
        sql_query = query_chain.invoke({
            "schema": schema,
            "user_query": user_input,
            "conversation_history": state["conversation_history"]
        })
        cleaned_query = clean_sql_query(sql_query.content)
        print("Generated SQL Query:", cleaned_query)  # Debugging output
        if "-- missing details for update" in cleaned_query.lower():
            state["update_pending"] = True
            state["sql_query"] = cleaned_query
        else:
            state["sql_query"] = cleaned_query
            vector_store.add_texts([state["sql_query"]])
    return state

def llm_call_node(state: State) -> State:
    """LLM Call Node: Processes response using LLM (for generic intents)."""
    if state["tool"] == "llm_response":
        generic_prompt = PromptTemplate(
            input_variables=["user_query", "conversation_history"],
            template="""
            Conversation history: {conversation_history}
            User asked: "{user_query}"

            As Ayman, created by Cenomi, I am a mall chatbot. Please ask me related to malls only.
            """
        )
        generic_chain = generic_prompt | llm | RunnablePassthrough()
        response = generic_chain.invoke({
            "user_query": state["user_input"],
            "conversation_history": state["conversation_history"]
        })
        state["llm_response"] = response.content.strip()
    return state

def response_generation_node(state: State) -> State:
    """Response Generation Node: Formats and prepares the output."""
    if state.get("sql_query"):
        # Check if the query indicates missing details for an update
        if state.get("update_pending") or ("update" in state.get("sub_intents", []) and len(state.get("sub_intents", [])) > 1):
            state["nlp_response"] = (
                "I need more details to update the offer. "
                "Please specify which offer you want to update (e.g., by title or ID) "
                "and what changes you want to make (e.g., new discount percentage, description, start date, or end date)."
            )
            if len(state.get("sub_intents", [])) > 1:
                state["nlp_response"] += " After that, I can handle your other request (e.g., listing offers) as needed."
        else:
            state["sql_results"], state["rows_affected"] = run_sql_query(state["sql_query"])
            # print("SQL Results:", state["sql_results"])  # Debugging output
            if not state["sql_results"] or "error" in json.loads(state["sql_results"]):
                retry_prompt = PromptTemplate(
                    input_variables=["schema", "user_query", "conversation_history", "previous_error"],
                    template="""
                    Database schema: {schema}
                    Conversation history: {conversation_history}
                    Previous query failed with error: {previous_error}
                    The user asked: "{user_query}"
                    Generate a corrected PostgreSQL query to fetch or update the required data from the database, using case-insensitive comparisons with ILIKE and LOWER().
                    If the user wants to update an offer but hasn't provided enough details, return a comment like:
                    -- Missing details for update. Please ask the user for the offer title or ID and the new values (e.g., discount percentage, start date, end date).
                    If the user specifies an offer by name (e.g., 'winter jacket'), assume it refers to an offer title like 'Winter Jacket Sale' if a close match exists in the context or schema. Update both discount_percentage and description if requested.
                    If the user requests multiple actions (e.g., update and list), prioritize the update action and generate a query or comment for it, noting the secondary action (e.g., list) will be handled after update details are provided.
                    If the user asks to 'list all the malls' or similar, generate a SELECT query to fetch all records from the malls table (e.g., SELECT mall_id, name, location FROM malls). If the schema does not explicitly list a malls table, assume a table named 'malls' with columns mall_id, name, and location exists.
                    If the user asks about customers or loyalty points, assume it is related to the mall's loyalty program and generate a query to fetch the relevant data from the appropriate table (e.g., customers table with loyalty_points column).
                    Otherwise, return only the SQL query without explanation.
                    """
                )
                retry_chain = retry_prompt | llm | RunnablePassthrough()
                previous_error = state["sql_results"] if state["sql_results"] else "No results returned"
                sql_query = retry_chain.invoke({
                    "schema": schema,
                    "user_query": state["user_input"],
                    "conversation_history": state["conversation_history"],
                    "previous_error": previous_error
                })
                state["sql_query"] = clean_sql_query(sql_query.content)
                if "-- missing details for update" in state["sql_query"].lower():
                    state["update_pending"] = True
                    state["nlp_response"] = (
                        "I need more details to update the offer. "
                        "Please specify which offer you want to update (e.g., by title or ID) "
                        "and what changes you want to make (e.g., new discount percentage, description, start date, or end date)."
                    )
                    if len(state.get("sub_intents", [])) > 1:
                        state["nlp_response"] += " After that, I can handle your other request (e.g., listing offers) as needed."
                else:
                    state["sql_results"], state["rows_affected"] = run_sql_query(state["sql_query"])
                    vector_store.add_texts([state["sql_query"]])

            if not state.get("nlp_response"):  # Only process if nlp_response isn't already set
                if not state["sql_results"] or "error" in json.loads(state["sql_results"]):
                    state["nlp_response"] = f"I'm sorry, I couldn't find the requested information in the database. The query attempted was: {state['sql_query']}. Schema used: {schema}"
                else:
                    try:
                        results_data = json.loads(state["sql_results"])
                        if isinstance(results_data, dict) and "rows_affected" in results_data:
                            rows_affected = results_data["rows_affected"]
                            user_input_lower = state["user_input"].lower()
                            offer_match = re.search(r"change\s+the\s+discount\s+on\s+(.+?)(?:\s+to|$)", user_input_lower)
                            offer_name = offer_match.group(1).strip() if offer_match else "offer"
                            percentage_match = re.search(r"to\s+(\d+)\s*percent", user_input_lower)
                            percentage = percentage_match.group(1) if percentage_match else "unknown"
                            description_match = re.search(r"description\s+to\s+(.+?)(?:\s+|$)", user_input_lower)
                            description = description_match.group(1).strip() if description_match else f"{percentage}% off {offer_name}"
                            if rows_affected > 0:
                                state["nlp_response"] = f"Okay, I've updated the {offer_name} to {percentage}% discount with description '{description}' (affected {rows_affected} offer{'s' if rows_affected > 1 else ''})."
                            else:
                                state["nlp_response"] = f"I couldn't find a {offer_name} to update. Please check the offer details and try again."
                        elif isinstance(results_data, list):
                            formatted_results = []
                            for row in results_data:
                                if "mall_id" in row and "name" in row:
                                    mall_info = f"Mall ID: {row['mall_id']}, Name: {row['name']}"
                                    if "location" in row:
                                        mall_info += f", Location: {row['location']}"
                                    formatted_results.append(mall_info)
                                elif "customer_id" in row and "first_name" in row and "last_name" in row and "loyalty_points" in row:
                                    customer_info = f"Customer ID: {row['customer_id']}, Name: {row['first_name']} {row['last_name']}, Loyalty Points: {row['loyalty_points']}"
                                    formatted_results.append(customer_info)
                                else:
                                    formatted_results.append(", ".join(f"{k}: {v}" for k, v in row.items()))
                            response_prompt = f"""
                            Conversation history: {state['conversation_history']}
                            User asked: "{state['user_input']}"
                            The database returned the following raw results: {formatted_results}
                            As Ayman, created by Cenomi, generate a clear and natural language response based on this data, including dates and times in a readable format. If the data contains mall information, list all malls. If it contains customer information, list all customers with their loyalty points.
                            """
                            state["nlp_response"] = llm.invoke(response_prompt).content.strip()
                        else:
                            state["nlp_response"] = "Unexpected response from the database."
                    except json.JSONDecodeError:
                        state["nlp_response"] = "Error processing database response."
    elif state.get("llm_response"):
        state["nlp_response"] = state["llm_response"]

    # Ensure nlp_response is set before updating memory and returning
    if "nlp_response" not in state:
        state["nlp_response"] = "An error occurred while processing your request. Please try again."

    update_memory(state["user_input"], state["nlp_response"])
    return state

def output_node(state: State) -> State:
    """Output Node: Sends response to chatbot interface."""
    if "nlp_response" not in state:
        state["output"] = json.dumps({"error": "Failed to generate response"}, indent=2)
    else:
        state["output"] = json.dumps({"message": state["nlp_response"]}, indent=2)
    return state