from langgraph.graph import StateGraph, END
from state import State
from nodes import user_input_node, retriever_node, context_builder_node, generator_node
import base64
import requests
def build_graph():
    builder = StateGraph(State)

    builder.add_node("user_input",      user_input_node)
    builder.add_node("retriever",       retriever_node)
    builder.add_node("context_builder", context_builder_node)
    builder.add_node("generator",       generator_node)

    builder.set_entry_point("user_input")

    builder.add_edge("user_input",      "retriever")
    builder.add_edge("retriever",       "context_builder")
    builder.add_edge("context_builder", "generator")
    builder.add_edge("generator",       END)

    return builder.compile()

rag_graph = build_graph()

print(rag_graph.get_graph().draw_ascii())

try:
    mermaid_text = rag_graph.get_graph().draw_mermaid()
    # Encode to base64
    graph_bytes = mermaid_text.encode("utf-8")
    encoded = base64.urlsafe_b64encode(graph_bytes).decode("utf-8")
    
    # Call Mermaid API to get PNG
    url = f"https://mermaid.ink/img/{encoded}"
    response = requests.get(url, timeout=10)
    
    if response.status_code == 200:
        with open("/app/downloads/graph.png", "wb") as f:
            f.write(response.content)
        print("[graph]  Graph image saved to downloads/graph.png")
    else:
        print(f"[graph] Failed to get image: {response.status_code}")
except Exception as e:
    print(f"[graph]  Error saving graph image: {e}")