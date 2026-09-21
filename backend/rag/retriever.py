from typing import Optional

from rag.embeddings import embedding_service
from rag.vector_store import vector_store
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    def __init__(self, top_k: int = None):
        self.top_k = top_k or settings.RAG_TOP_K

    def retrieve(self, query: str) -> str:
        if not vector_store.is_configured:
            return ""

        try:
            query_embedding = embedding_service.embed_text(query)
            results = vector_store.query(query_embedding, top_k=self.top_k)

            if not results:
                logger.info("RAG: no relevant chunks found")
                return ""

            context = "\n\n".join(
                f"[Source: {r['source']}]\n{r['text']}"
                for r in results
            )
            logger.info(
                f"RAG: retrieved {len(results)} chunks "
                f"(top score: {results[0]['score']:.3f})"
            )
            return context

        except Exception as e:

            logger.error(f"RAG retrieval failed: {e}")
            return ""

retriever = Retriever()