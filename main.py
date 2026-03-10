from dotenv import load_dotenv
load_dotenv()  # MUST be first — loads .env before anything else

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils import download_pdf, extract_text_from_pdf, chunk_text
from retriever import create_embeddings, init_pinecone, upsert_chunks
from nodes import set_embedding_model
from graph import rag_graph
from state import State


ML_PDF_URL = "https://alexjungaalto.github.io/MLBasicsBook.pdf"


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ML Basics RAG API",
    description=(
        "Ask any question about the ML Basics textbook.\n\n"
        "**Pipeline:** User Input → Retriever → Context Builder → Generator (Groq LLM)"
    ),
    version="1.0.0"
)


# ── Startup: runs once when container starts ──────────────────────────────────
@app.on_event("startup")
def startup():
    print("=" * 55)
    print(" Starting ML RAG Pipeline...")
    print("=" * 55)

    # Step 1: Download ML Basics PDF
    pdf_path = download_pdf(ML_PDF_URL, save_dir="downloads")

    # Step 2: Extract text + chunk it
    text   = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(
        text,
        chunk_size=int(os.getenv("CHUNK_SIZE", 500)),
        overlap=int(os.getenv("CHUNK_OVERLAP", 100))
    )
    print(f"[main] PDF split into {len(chunks)} chunks.")

    # Step 3: Create embeddings (MiniLM, runs locally, no API key)
    print("[main] Creating embeddings...")
    chunk_embeddings, embedding_model = create_embeddings(chunks)
    print(f"[main] Embeddings shape: {chunk_embeddings.shape}")

    # Step 4: Connect to Pinecone + upsert all vectors
    index = init_pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name=os.getenv("PINECONE_INDEX_NAME", "ml-rag-index")
    )
    upsert_chunks(index, chunks, chunk_embeddings)

    # Step 5: Share embedding model with nodes (used at query time)
    set_embedding_model(embedding_model)

    print("[main] Ready! Visit http://localhost:8001/docs")
    print("=" * 55)


# ── Request / Response schemas ────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {"question": "What is gradient descent?"}
        }

class AnswerResponse(BaseModel):
    question: str
    answer:   str


# ── POST /ask — main RAG endpoint ─────────────────────────────────────────────
@app.post("/ask", response_model=AnswerResponse, summary="Ask a question about ML")
def ask(request: QuestionRequest):
    """
    **Workflow:**
    1. User sends a question
    2. Retriever searches Pinecone for relevant ML textbook chunks
    3. Context Builder combines question + chunks into a prompt
    4. Generator (Groq LLM) produces the answer
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Initialize state — user input node
    initial_state: State = {
        "question":  request.question,
        "documents": [],
        "context":   "",
        "answer":    ""
    }

    # Run full LangGraph pipeline
    result = rag_graph.invoke(initial_state)

    if not result.get("answer"):
        raise HTTPException(status_code=500, detail="LLM did not return an answer.")

    return AnswerResponse(
        question=request.question,
        answer=result["answer"]
    )


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "message": "ML RAG API is running"}


# ── Local run (outside Docker) ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)