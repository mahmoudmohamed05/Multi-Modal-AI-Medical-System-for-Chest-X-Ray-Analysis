from functools import lru_cache
import chromadb
from app.config import get_settings


@lru_cache
def get_collection():
    settings = get_settings()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(ids: list[str], texts: list[str], metadatas: list[dict], embeddings: list[list[float]]):
    if not ids:
        return
    get_collection().upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def semantic_search(query_embedding: list[float], top_k: int) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    rows = []
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for doc_id, doc, meta, distance in zip(ids, docs, metas, distances):
        rows.append({
            "id": doc_id, "text": doc, "metadata": meta or {},
            "distance": float(distance),
        })
    return rows
