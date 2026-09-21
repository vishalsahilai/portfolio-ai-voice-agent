from typing import List, Optional

from pinecone import Pinecone, ServerlessSpec

from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


class PineconeVectorStore:
    def __init__(
        self,
        api_key: str = None,
        index_name: str = None,
        cloud: str = None,
        region: str = None,
        dimension: int = None,
    ):
        self.api_key = api_key or settings.PINECONE_API_KEY
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.cloud = cloud or settings.PINECONE_CLOUD
        self.region = region or settings.PINECONE_REGION
        self.dimension = dimension or settings.EMBEDDING_DIMENSION

        self.is_configured = bool(self.api_key)

        if not self.is_configured:
            logger.warning(
                "No PINECONE_API_KEY configured — "
                "RAG retrieval will be skipped until .env is filled in"
            )

        self._pc: Optional[Pinecone] = (
            Pinecone(api_key=self.api_key)
            if self.is_configured
            else None
        )

        self._index = None

    def _ensure_index(self) -> None:
        if self._index is not None:
            return

        if not self.is_configured:
            raise RuntimeError(
                "PineconeVectorStore has no API key configured"
            )

        if not self._pc.has_index(self.index_name):
            logger.info(
                f"Pinecone index '{self.index_name}' "
                f"doesn't exist yet — creating it"
            )

            self._pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.cloud,
                    region=self.region,
                ),
            )

        self._index = self._pc.Index(self.index_name)

    def upsert_chunks(
        self,
        chunks: List[dict],
    ) -> None:
        """
        Upload the current document chunks into Pinecone.

        Expected chunk structure:
        {
            "id": str,
            "embedding": List[float],
            "text": str,
            "source": str,
            "source_id": str,
            "version": str,
            "chunk_index": int,
        }
        """

        if not chunks:
            return

        self._ensure_index()

        vectors = [
            {
                "id": chunk["id"],
                "values": chunk["embedding"],
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "source_id": chunk["source_id"],
                    "version": chunk["version"],
                    "chunk_index": chunk["chunk_index"],
                },
            }
            for chunk in chunks
        ]

        self._index.upsert(
            vectors=vectors
        )

        logger.info(
            f"Upserted {len(vectors)} chunks into "
            f"Pinecone index '{self.index_name}'"
        )

    def delete_old_document_versions(
        self,
        source_id: str,
        current_version: str,
    ) -> None:
        """
        Delete all previous versions of a document while
        keeping only the current version.
        """

        self._ensure_index()

        self._index.delete(
            filter={
                "$and": [
                    {
                        "source_id": {
                            "$eq": source_id
                        }
                    },
                    {
                        "version": {
                            "$ne": current_version
                        }
                    },
                ]
            }
        )

        logger.info(
            f"Removed old versions for "
            f"source_id='{source_id}'. "
            f"Keeping version='{current_version}'"
        )

    def query(
        self,
        embedding: List[float],
        top_k: int = 3,
    ) -> List[dict]:
        """
        Search Pinecone for the most relevant
        knowledge chunks.
        """

        if not self.is_configured:
            return []

        self._ensure_index()

        result = self._index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
        )

        return [
            {
                "text": match.metadata.get(
                    "text",
                    "",
                ),
                "source": match.metadata.get(
                    "source",
                    "",
                ),
                "score": match.score,
            }
            for match in result.matches
        ]


vector_store = PineconeVectorStore()