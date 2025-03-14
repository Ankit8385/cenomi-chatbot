from langgraph.graph import StateGraph
from config import State  # Import State from config.py
from nodes import input_node, memory_node, intent_router_node, tool_selection_node, tool_invocation_node, llm_call_node, response_generation_node, output_node

# Define the workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("input", input_node)
workflow.add_node("memory", memory_node)
workflow.add_node("intent_router", intent_router_node)
workflow.add_node("tool_selection", tool_selection_node)
workflow.add_node("tool_invocation", tool_invocation_node)
workflow.add_node("llm_call", llm_call_node)
workflow.add_node("response_generation", response_generation_node)
workflow.add_node("output", output_node)

# Define edges (sequential flow)
workflow.add_edge("input", "memory")
workflow.add_edge("memory", "intent_router")
workflow.add_edge("intent_router", "tool_selection")
workflow.add_edge("tool_selection", "tool_invocation")
workflow.add_edge("tool_invocation", "llm_call")
workflow.add_edge("llm_call", "response_generation")
workflow.add_edge("response_generation", "output")

# Set entry and finish points
workflow.set_entry_point("input")
workflow.set_finish_point("output")

# Compile the workflow
app = workflow.compile()

# Main loop
if __name__ == "__main__":
    while True:
        user_input = input("Ask your query (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        initial_state = {"user_input": user_input}
        result = app.invoke(initial_state)
        print("\nChatbot Response:\n", result["output"])