import json
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings


def load_knowledge_export() -> int:
    export_path = Path(__file__).resolve().parents[2] / "data" / "knowledge_export.json"

    if not export_path.exists():
        print(f"Knowledge export not found: {export_path}")
        return 0

    pointer_path = Path(settings.chroma_persist_directory) / "active_collection.txt"
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_directory,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # Keep existing active knowledge when it is already populated.
    if pointer_path.exists():
        collection_name = pointer_path.read_text(encoding="utf-8").strip()
        if collection_name:
            try:
                existing = client.get_collection(collection_name)
                if existing.count() > 0:
                    print(f"Existing knowledge collection preserved: {collection_name} ({existing.count()} chunks)")
                    return existing.count()
            except Exception:
                pass

    records = json.loads(export_path.read_text(encoding="utf-8"))
    if not records:
        print("Knowledge export is empty; no data imported.")
        return 0

    collection_name = "placidway_bundled_knowledge"
    target = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if target.count() == 0:
        batch_size = 100
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            ids = [f"bundled_{start + i}" for i in range(len(batch))]
            documents = [item["text"] for item in batch]
            metadatas = [item.get("metadata") or {} for item in batch]
            embeddings = [item["embedding"] for item in batch]

            target.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

    if target.count() > 0:
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        temp_pointer = pointer_path.with_suffix(".tmp")
        temp_pointer.write_text(collection_name, encoding="utf-8")
        temp_pointer.replace(pointer_path)

    print(f"Bundled knowledge loaded: {target.count()} chunks")
    return target.count()
