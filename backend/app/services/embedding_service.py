from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.config import settings

@lru_cache(maxsize=1)
def model():        
    return SentenceTransformer(settings.embedding_model)

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = model().encode(texts, normalize_embeddings=True)
    return vectors.tolist()

def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
