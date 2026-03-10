from state import State
from llm import llm
from retriever import retrieve_relevant_docs
from langchain_core.messages import HumanMessage

_embedding_model = None

def set_embedding_model(model):
    global _embedding_model
    _embedding_model = model


# NODE 1 — USER INPUT
def user_input_node(state: State) -> State:
    question = state["question"].strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    state["question"] = question
    print(f"[user_input_node] Question received: {question}")
    return state


# NODE 2 — RETRIEVER
def retriever_node(state: State) -> State:
    print(f"[retriever_node] Searching Pinecone for: {state['question']}")
    docs = retrieve_relevant_docs(state["question"], _embedding_model)
    state["documents"] = docs
    print(f"[retriever_node] Found {len(docs)} chunks.")
    return state


# NODE 3 — CONTEXT BUILDER
def context_builder_node(state: State) -> State:
    question  = state["question"]
    documents = state["documents"]

    chunks_text = "\n\n".join([
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in documents
    ])

    context = (
        f"User Question:\n{question}\n\n"
        f"Relevant Knowledge from ML Textbook:\n{chunks_text}"
    )

    state["context"] = context
    print("[context_builder_node] Context built.")
    return state


# NODE 4 — GENERATOR
def generator_node(state: State) -> State:
    prompt = (
        "You are a helpful Machine Learning tutor.\n"
        "A student has asked you a question. "
        "Using the textbook context provided below, write a clear, complete, "
        "and easy-to-understand answer in full sentences. "
        "Do NOT just copy the context — explain it properly.\n\n"
        f"Context from ML Textbook:\n{state['context']}\n\n"
        f"Question: {state['question']}\n\n"
        "Answer in full sentences:"
    )

    print("[generator_node] Calling Groq LLM ...")
    response = llm.invoke([HumanMessage(content=prompt)])
    state["answer"] = response.content
    print("[generator_node] Answer generated.")
    return state