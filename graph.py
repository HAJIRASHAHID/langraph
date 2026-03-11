from langgraph.graph import StateGraph, END
from state import State
from nodes import user_input_node, retriever_node, tavily_search_node, context_builder_node, generator_node

# This function DECIDES which path to take
def should_web_search(state: State) -> str:
    if state["documents"]:
        print("[graph] Docs found → Pinecone")
        return "context_builder"
    else:
        print("[graph] No docs → Web Search ")
        return "tavily_search"


def build_graph():
    builder = StateGraph(State)

    # Register all nodes
    builder.add_node("user_input",      user_input_node)
    builder.add_node("retriever",       retriever_node)
    builder.add_node("tavily_search",   tavily_search_node)  # 👈 add this
    builder.add_node("context_builder", context_builder_node)
    builder.add_node("generator",       generator_node)

    # Wire the flow
    builder.set_entry_point("user_input")
    builder.add_edge("user_input", "retriever")

    # After retriever — decide which path
    builder.add_conditional_edges(
        "retriever",          # from this node
        should_web_search,    # use this function to decide
        {
            "context_builder": "context_builder",  # if docs found
            "tavily_search": "tavily_search"             # if no docs
        }
    )

    # Both paths join here
    builder.add_edge("tavily_search",      "context_builder")
    builder.add_edge("context_builder", "generator")
    builder.add_edge("generator",       END)

    return builder.compile()


rag_graph = build_graph()