from chromadb.config import Settings as ChromaSettings
from pathlib import Path
from app.config import settings
from app.services.embedding_service import embed_query
from app.services.store import collection

def active_collection():
    pointer = Path(settings.chroma_persist_directory) / "active_collection.txt"
    if pointer.exists():
        import chromadb
        client = chromadb.PersistentClient(
    path=settings.chroma_persist_directory,
    settings=ChromaSettings(anonymized_telemetry=False),
)
        return client.get_collection(pointer.read_text(encoding="utf-8").strip())
    return collection()


def retrieve(question: str, top_k: int | None = None):
    col = active_collection()
    if col.count() == 0:
        return []

    result = col.query(
        query_embeddings=[embed_query(question)],
        n_results=min(top_k or settings.top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    found = []
    for doc, meta, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        score = 1 - float(distance) / 2

        if score < settings.min_retrieval_score:
            continue

        metadata = meta or {}
        found.append({
            "text": doc,
            "metadata": metadata,
            "source_url": metadata.get("url") or metadata.get("source_url"),
            "source_title": metadata.get("title") or metadata.get("source_title"),
            "score": score,
        })

    return found
