import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings

def collection():
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_directory,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name="placidway_approved_pages",
        metadata={"hnsw:space": "cosine"},
    )
