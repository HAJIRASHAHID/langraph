import os
import requests
import fitz  # PyMuPDF


def download_pdf(url: str, save_dir: str = "downloads") -> str:
    """Download PDF from URL. Skips if already downloaded."""
    os.makedirs(save_dir, exist_ok=True)
    filename = url.split("/")[-1]
    pdf_path = os.path.join(save_dir, filename)

    if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
        print(f"[utils] Downloading PDF from {url} ...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(pdf_path, "wb") as f:
            f.write(r.content)
        print(f"[utils] Saved to {pdf_path}")
    else:
        print(f"[utils] PDF already exists at {pdf_path}, skipping.")

    return pdf_path


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc if page.get_text().strip()]
    doc.close()
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks