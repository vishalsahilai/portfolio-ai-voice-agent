import glob
import hashlib
import os
from typing import List

import pymupdf

from rag.embeddings import embedding_service
from rag.vector_store import vector_store
from utils.logger import get_logger


logger = get_logger(__name__)


KNOWLEDGE_BASE_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(__file__)
    ),
    "knowledge_base",
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def get_file_version(
    file_path: str,
) -> str:
    """
    Generate a version hash from the actual file contents.

    Same file contents:
        same version

    Changed file contents:
        new version
    """

    hasher = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            hasher.update(block)

    return hasher.hexdigest()[:16]


def get_source_id(
    filename: str,
) -> str:
    """
    Generate a stable document identifier.

    Keep the same filename when updating/replacing
    the same knowledge document.
    """

    return filename.lower()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:

    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(
                chunk
            )

        start = end - overlap

    return chunks


def load_documents() -> List[dict]:
    """
    Load supported knowledge documents from
    the knowledge_base directory.

    Supported:
    - .txt
    - .md
    - .pdf
    """

    paths = (
        glob.glob(
            os.path.join(
                KNOWLEDGE_BASE_DIR,
                "*.txt",
            )
        )
        + glob.glob(
            os.path.join(
                KNOWLEDGE_BASE_DIR,
                "*.md",
            )
        )
        + glob.glob(
            os.path.join(
                KNOWLEDGE_BASE_DIR,
                "*.pdf",
            )
        )
    )

    documents = []

    for path in paths:

        source = os.path.basename(
            path
        )

        source_id = get_source_id(
            source
        )

        version = get_file_version(
            path
        )

        if path.lower().endswith(".pdf"):

            pdf = pymupdf.open(
                path
            )

            try:
                text = "\n".join(
                    page.get_text()
                    for page in pdf
                )
            finally:
                pdf.close()

        else:

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as file:
                text = file.read()

        if not text.strip():

            logger.warning(
                f"Document '{source}' "
                f"has no readable text — skipping"
            )

            continue

        documents.append(
            {
                "source": source,
                "source_id": source_id,
                "version": version,
                "text": text,
            }
        )

        logger.info(
            f"Loaded document "
            f"'{source}' "
            f"version={version}"
        )

    return documents


def run_ingestion() -> None:

    documents = load_documents()

    if not documents:

        logger.warning(
            f"No knowledge documents found in "
            f"{KNOWLEDGE_BASE_DIR}"
        )

        return

    all_chunks = []

    for document in documents:

        pieces = chunk_text(
            document["text"]
        )

        logger.info(
            f"Document "
            f"'{document['source']}' "
            f"split into "
            f"{len(pieces)} chunks"
        )

        for index, piece in enumerate(
            pieces
        ):

            vector_id = (
                f"{document['source_id']}:"
                f"{document['version']}:"
                f"{index}"
            )

            all_chunks.append(
                {
                    "id": vector_id,
                    "text": piece,
                    "source": document[
                        "source"
                    ],
                    "source_id": document[
                        "source_id"
                    ],
                    "version": document[
                        "version"
                    ],
                    "chunk_index": index,
                }
            )

    if not all_chunks:

        logger.warning(
            "Knowledge documents produced "
            "no chunks — nothing to ingest"
        )

        return

    logger.info(
        f"Created {len(all_chunks)} chunks "
        f"from {len(documents)} document(s)"
    )

    logger.info(
        "Generating embeddings..."
    )

    embeddings = (
        embedding_service.embed_batch(
            [
                chunk["text"]
                for chunk in all_chunks
            ]
        )
    )

    if len(embeddings) != len(
        all_chunks
    ):
        raise RuntimeError(
            "Embedding count does not match "
            "chunk count. Existing Pinecone "
            "data was not modified."
        )

    for chunk, embedding in zip(
        all_chunks,
        embeddings,
    ):
        chunk["embedding"] = embedding

    logger.info(
        "Embeddings generated successfully"
    )

    # Upload the newest document versions first.
    #
    # This protects the existing knowledge base.
    # If embedding generation or upload fails,
    # older working data remains available.
    vector_store.upsert_chunks(
        all_chunks
    )

    logger.info(
        "Current document versions "
        "uploaded successfully"
    )

    # After successful upload, remove all
    # previous versions of each document.
    for document in documents:

        vector_store.delete_old_document_versions(
            source_id=document[
                "source_id"
            ],
            current_version=document[
                "version"
            ],
        )

    logger.info(
        "Ingestion complete — Pinecone now "
        "contains only the latest version "
        "of each ingested document"
    )


if __name__ == "__main__":
    run_ingestion()