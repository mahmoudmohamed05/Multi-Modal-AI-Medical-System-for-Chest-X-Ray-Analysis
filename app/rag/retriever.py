from app.config import get_settings
from app.rag.embeddings import embed_query
from app.rag.vector_store import semantic_search
from app.rag.bm25_store import bm25_search


def hybrid_search(query: str, top_k: int | None = None) -> list[dict]:
    settings = get_settings()
    final_k = top_k or settings.final_top_k
    vector_rows = semantic_search(embed_query(query), settings.vector_top_k)
    keyword_rows = bm25_search(query, settings.bm25_top_k)

    # Reciprocal Rank Fusion (RRF): fuse ranks, not raw scores from different systems.
    fused: dict[str, dict] = {}
    def add_ranked(rows: list[dict], key: str):
        for rank, row in enumerate(rows, start=1):
            doc_id = row.get("id") or row.get("chunk_id")
            if not doc_id:
                continue
            if doc_id not in fused:
                fused[doc_id] = {
                    "id": doc_id, "text": row["text"],
                    "metadata": row.get("metadata", {}),
                    "rrf_score": 0.0,
                }
            fused[doc_id]["rrf_score"] += 1.0 / (settings.rrf_k + rank)
            fused[doc_id][key] = rank

    add_ranked(vector_rows, "vector_rank")
    add_ranked(keyword_rows, "bm25_rank")
    return sorted(fused.values(), key=lambda row: row["rrf_score"], reverse=True)[:final_k]
