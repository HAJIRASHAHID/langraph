from state import State
from llm import llm
from retriever import retrieve_relevant_docs
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from tavily import TavilyClient
import os

# ------------------------------------------------
# Global embedding model (set at startup)
# ------------------------------------------------
_embedding_model = None


def set_embedding_model(model):
    global _embedding_model
    _embedding_model = model


def get_embedding_model():
    if _embedding_model is None:
        raise ValueError("[nodes] Embedding model not initialized!")
    return _embedding_model


# ------------------------------------------------
# Tool: Tavily Web Search
# ------------------------------------------------
@tool  # decorator to make it a callable tool in LangGraph and LLMs ... but we will call it directly too
def tavily_search_tool(query: str) -> str:
    """Search the web for information when the knowledge base is insufficient to answer the question."""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query=query, max_results=5)
    print(f"[tavily_search_tool] Search results for query: '{query}'")
    return "\n\n".join([r["content"] for r in results.get("results", [])])


# LLM with tool bound — this is the ONLY LLM used
llm_with_tools = llm.bind_tools(
    [tavily_search_tool]
)  # bind tools tells the LLM which tools it can call, so it can decide when to call them based on the prompt and question


# ------------------------------------------------
# USER INPUT NODE
# ------------------------------------------------
def user_input_node(state: State) -> State:
    question = state["question"].strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    state["question"] = question
    return state


# ------------------------------------------------
# RETRIEVER NODE
# ------------------------------------------------
def retriever_node(state: State) -> State:
    embedding_model = get_embedding_model()
    docs = retrieve_relevant_docs(state["question"], embedding_model)
    state["documents"] = docs
    return state


# ------------------------------------------------
# CONTEXT BUILDER NODE
# ------------------------------------------------
def context_builder_node(state: State) -> State:
    question = state["question"]
    documents = state.get("documents", [])

    if documents:
        chunks_text = "\n\n".join(documents)
        source = "ML Textbook"
    else:
        chunks_text = "No relevant documents found in Pinecone."
        source = "None"

    state["context"] = f"User Question:\n{question}\n\n" f"Relevant Knowledge from {source}:\n{chunks_text}"
    return state


# ------------------------------------------------
# GENERATOR NODE — LLM decides to call tool or not
# ------------------------------------------------
def generator_node(state: State) -> State:
    question = state["question"]

    # ml_keywords = [
    #     "machine learning",     not required simple way by prompt
    #     "ml",
    #     "gradient",
    #     "neural",
    #     "model",
    #     "training",
    #     "dataset",
    #     "algorithm",
    #     "loss",
    #     "feature",
    #     "classification",
    #     "regression",
    #     "supervised",
    #     "unsupervised",
    #     "deep learning",
    #     "optimization",
    #     "vector",
    #     "matrix",
    # ]

    # question_lower = question.lower()
    # is_ml_question = any(kw in question_lower for kw in ml_keywords)

    # if is_ml_question:
    # Extract only clean sentences from context, skip lines with math symbols
    raw_chunks = state.get("documents", [])
    clean_lines = []
    for chunk in raw_chunks:
        for line in chunk.split("\n"):
            line = line.strip()
            # Skip lines that are mostly math/symbols/numbers
            if len(line) > 40 and sum(c.isalpha() for c in line) > len(line) * 0.5:
                clean_lines.append(line)
    clean_context = "\n".join(clean_lines[:20])  # top 20 clean lines only

    prompt = f"""You are a helpful Machine Learning tutor.

        If the user query can be answered based on the provided context from the ML textbook, answer using only that information. Do NOT use any information from your training data or memory, as it may be outdated or incorrect. If not then you MUST call the tavily_search_tool to search the web for the latest and most accurate answer. Make sure to always call the tavily search tool for user queries that are not related to machine learning.
        Write a clean, friendly explanation in your own words like a teacher.
        Do NOT copy any text. No formulas, no symbols, no raw math.

        Key ideas from textbook:
        {clean_context}

        Question:
        {question}

        Answer in clear, full sentences only."""

    # else:
    #     prompt = f"""You are a helpful assistant.

    #     This question is NOT related to Machine Learning.
    #     You MUST call the tavily_search_tool to search the web for the latest and most accurate answer.
    #     Do NOT answer from memory or training data as it may be outdated.

    #     Question:
    #     {question}"""

    messages = [HumanMessage(content=prompt)]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_result = tavily_search_tool.invoke(tool_call["args"])
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
        final_response = llm.invoke(messages)
        state["answer"] = final_response.content

    elif response.content and response.content.strip():
        state["answer"] = response.content

    else:
        final_response = llm.invoke([HumanMessage(content=prompt)])
        state["answer"] = final_response.content

    return state