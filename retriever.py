from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX_NAME", "ml-rag-index")


# ── Step 1: Create embeddings from chunks
def create_embeddings(chunks: list, batch_size: int = 64):
    """
    Loads MiniLM locally, encodes all chunks in batches.
    Returns:
        chunk_embeddings → numpy array of float32 vectors (shape: N x 384)
        embedding_model  → kept alive for query-time embedding
    """
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = embedding_model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return np.array(chunk_embeddings).astype("float32"), embedding_model


# ── Step 2: Connect to Pinecone index 
def init_pinecone(api_key: str, index_name: str):
    """
    Connects to Pinecone and returns the index object.
    Creates the index if it doesn't exist (dim=384 for MiniLM).
    """
    try:
        pc = Pinecone(api_key=api_key)

        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            print(f"[retriever] Creating Pinecone index '{index_name}' ...")
            pc.create_index(
                name=index_name,
                dimension=384,       # all-MiniLM-L6-v2 output size
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print("[retriever] Index created.")
        else:
            print(f"[retriever] Index '{index_name}' already exists.")

        return pc.Index(index_name)

    except Exception as e:
        print(f"[retriever] Pinecone error: {e}")
        return None


# ── Step 3: Upsert chunks + their vectors into Pinecone
def upsert_chunks(index, chunks: list, chunk_embeddings, batch_size: int = 50):
    """
    Uploads chunks to Pinecone in batches.
    Each vector stores the original text in metadata for retrieval.
    """
    if chunk_embeddings is None or len(chunk_embeddings) == 0:
        print("[retriever] No embeddings to upsert.")
        return

    total = len(chunk_embeddings)
    print(f"[retriever] Upserting {total} vectors to Pinecone...")

    for i in range(0, total, batch_size):
        batch_vectors = []
        for j in range(i, min(i + batch_size, total)):
            emb = chunk_embeddings[j]
            batch_vectors.append({
                "id":       str(j),
                "values":   emb.tolist(),
                "metadata": {"text": chunks[j]}   # store original text
            })
        index.upsert(vectors=batch_vectors)

    print("[retriever] Upsert complete.")


# ── Step 4: Retrieve relevant chunks at query time 
def retrieve_relevant_docs(question: str, embedding_model: SentenceTransformer, k: int = 3) -> list:
    """
    Embeds the user question → queries Pinecone → returns top-k text chunks.
    """
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    # Embed the question using the same model used at ingestion
    question_vector = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )[0].astype("float32").tolist()

    # Search Pinecone for most similar chunks
    results = index.query(
        vector=question_vector,
        top_k=k,
        include_metadata=True   # get back the original text
    )

    # Extract text from metadata
    docs = [match["metadata"]["text"] for match in results["matches"]]
    print(f"[retriever] Retrieved {len(docs)} chunks for query.")
    return docs