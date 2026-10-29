import json
from pathlib import Path
from app.services.rag_service import active_collection

col = active_collection()
data = col.get(include=["documents", "metadatas", "embeddings"])

records = []
for i, doc in enumerate(data["documents"]):
    records.append({
        "text": doc,
        "metadata": data["metadatas"][i] or {},
        "embedding": (
            data["embeddings"][i].tolist()
            if data["embeddings"] is not None
            else None
        ),
    })

output = Path("data/knowledge_export.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(records, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Exported chunks: {len(records)}")
print(f"Output file: {output.resolve()}")
print(f"File size: {output.stat().st_size / 1024:.1f} KB")
