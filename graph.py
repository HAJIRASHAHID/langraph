from langgraph.graph import StateGraph, END
from state import State
from nodes import user_input_node, retriever_node, context_builder_node, generator_node


def build_graph():

 
    builder = StateGraph(State)

    # Register all 4 nodes
    builder.add_node("user_input",      user_input_node)
    builder.add_node("retriever",       retriever_node)
    builder.add_node("context_builder", context_builder_node)
    builder.add_node("generator",       generator_node)

    # Wire the flow
    builder.set_entry_point("user_input")
    builder.add_edge("user_input",      "retriever")
    builder.add_edge("retriever",       "context_builder")
    builder.add_edge("context_builder", "generator")
    builder.add_edge("generator",       END)

    return builder.compile()


rag_graph = build_graph()