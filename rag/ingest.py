import glob
import hashlib
import os
from typing import List
 
from rag.embeddings import embedding_service
from rag.vector_store import vector_store
from utils.logger import get_logger
 
logger = get_logger(__name__)

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
CHUNK_SIZE = 800     
CHUNK_OVERLAP = 100   

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []
 
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap 
    return [c for c in chunks if c]
 
def load_documents() -> List[dict]:
    """Reads every .txt/.md file in knowledge_base/. Returns [{"source": filename, "text": full_text}, ...]."""
    paths = (
        glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.txt")) +
        glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.md")) +
        glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.pdf"))
    )
    documents = []
    for path in paths:
        if path.endswith(".pdf"):
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            text = "\n".join(page.get_text() for page in doc)
        else:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        documents.append({"source": os.path.basename(path), "text": text})
    return documents
 
def run_ingestion() -> None:
    documents = load_documents()
    if not documents:
        logger.warning(f"No knowledge documents found in {KNOWLEDGE_BASE_DIR} — nothing to ingest")
        return
 
    all_chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunk_hash = hashlib.sha1(piece.encode("utf-8")).hexdigest()[:8]
            all_chunks.append({
                "id": f"{doc['source']}-{i}-{chunk_hash}",
                "text": piece,
                "source": doc["source"],
            })
 
    logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks, embedding...")
    embeddings = embedding_service.embed_batch([c["text"] for c in all_chunks])
    for chunk, embedding in zip(all_chunks, embeddings):
        chunk["embedding"] = embedding
 
    vector_store.upsert_chunks(all_chunks)
    logger.info("Ingestion complete")
 
 
if __name__ == "__main__":
    run_ingestion()