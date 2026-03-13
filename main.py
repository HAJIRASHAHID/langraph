from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from utils import download_pdf, extract_text_from_pdf, chunk_text
from retriever import upsert_chunks
from sentence_transformers import SentenceTransformer
from nodes import set_embedding_model
from graph import rag_graph
from state import State

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "langraph-rag"

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

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    stats = index.describe_index_stats()
    if stats.get("total_vector_count", 0) > 0:
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