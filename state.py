# State = shared memory between all LangGraph nodes
# Every node reads from and writes to this dictionary

from typing import TypedDict, List

class State(TypedDict):
    question:  str        # Node: User Input  → set by FastAPI endpoint
    documents: list       # Node: Retriever   → top-k chunks from Pinecone
    context:   str        # Node: Context Builder → combined prompt context
    answer:    str        # Node: Generator   → final LLM answer
    web_results: str       # Node: results from tavily ....not used 
