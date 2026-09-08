from typing import List

from pinecone import Pinecone

from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self):
        if not settings.PINECONE_API_KEY:
            raise RuntimeError(
                "PINECONE_API_KEY is not configured"
            )

        self.model = settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION

        self.client = Pinecone(
            api_key=settings.PINECONE_API_KEY
        )

    def _embed(
        self,
        texts: List[str],
        input_type: str,
    ) -> List[List[float]]:
        if not texts:
            return []

        response = self.client.inference.embed(
            model=self.model,
            inputs=texts,
            parameters={
                "input_type": input_type,
                "truncate": "END",
                "dimension": self.dimension,
            },
        )

        return [
            list(item.values)
            for item in response.data
        ]

    def embed_text(
        self,
        text: str,
    ) -> List[float]:
        if not text.strip():
            return []

        return self._embed(
            [text],
            "query",
        )[0]

    def embed_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        return self._embed(
            texts,
            "passage",
        )


embedding_service = EmbeddingService()