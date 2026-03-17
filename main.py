from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from pinecone import Pinecone, ServerlessSpec
from utils import download_pdf, extract_text_from_pdf, chunk_text
from retriever import upsert_chunks
from sentence_transformers import SentenceTransformer
from nodes import set_embedding_model
from graph import rag_graph
from state import State

os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "langraph-rag"

ML_PDF_URL = "https://alexjungaalto.github.io/MLBasicsBook.pdf"

app = FastAPI(
    title="ML Basics RAG API",
    description="Ask questions about the ML Basics textbook.",
    version="1.0.0",
)

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
set_embedding_model(embedding_model)

# Startup: download PDF, chunk, embed, upload
@app.on_event("startup")
async def startup_event():
    print("=======================================================")
    print(" Starting ML RAG Pipeline...")
    print("=======================================================")

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY")) #initialize Pinecone client
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    stats = index.describe_index_stats()
    if stats.get("total_vector_count", 0) > 0: # If Pinecone already has vectors, skip embedding creation to save time during development
        print("[main] Pinecone already contains vectors. Skipping embedding creation.")
        return

    pdf_path = download_pdf(ML_PDF_URL)
    chunks = chunk_text(extract_text_from_pdf(pdf_path))
    print(f"[main] PDF split into {len(chunks)} chunks.")

    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    upsert_chunks(index, chunks, embeddings)
    print("[main] Ready!")

# API Schemas
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

# Main endpoint
@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    initial_state: State = {
        "question": request.question,
        "documents": [],
        "context": "",
        "answer": "",
    }

    try:
        result = rag_graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {e}")

    return AnswerResponse(
        question=request.question,
        answer=result.get("answer", "No answer generated.")
    )

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tracing")
def get_tracing_graph():
    graph_path = "/app/downloads/graph.png"  # absolute path inside Docker
    if os.path.exists(graph_path):
        return FileResponse(graph_path, media_type="image/png")
    return {"error": "Graph not generated yet"}