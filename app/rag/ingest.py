import hashlib
from app.config import get_settings
from app.rag.document_loader import load_documents, chunk_text
from app.rag.embeddings import embed_texts
from app.rag.vectore_store import upsert_chunks
from app.rag.bm25_store import save_corpus


def _chunk_id(source: str, page: int | None, index: int, text: str) -> str:
    raw = f"{source}|{page}|{index}|{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def ingest_directory() -> int:
    settings = get_settings()
    documents = load_documents(settings.documents_dir)
    if not documents:
        raise RuntimeError(
            f"No PDF, TXT or Markdown text files found in {settings.documents_dir.resolve()}"
        )

    ids, texts, metadatas, bm25_rows = [], [], [], []
    for doc in documents:
        for index, chunk in enumerate(
            chunk_text(doc["text"], settings.chunk_size, settings.chunk_overlap)
        ):
            chunk_id = _chunk_id(doc["source"], doc["page"], index, chunk)
            metadata = {
                "source": doc["source"],
                "page": doc["page"] if doc["page"] is not None else 0,
                "chunk_index": index,
                "source_path": doc["source_path"],
            }
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append(metadata)
            bm25_rows.append({
                "id": chunk_id, "chunk_id": chunk_id, "text": chunk,
                "metadata": metadata,
            })

    # Embed in batches to avoid high memory use on larger corpora.
    batch_size = 64
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        vectors = embed_texts(texts[start:end])
        upsert_chunks(ids[start:end], texts[start:end], metadatas[start:end], vectors)

    save_corpus(bm25_rows)
    print(f"Ingested/upserted {len(texts)} chunks from {len(documents)} document pages/files.")
    return len(texts)


if __name__ == "__main__":
    ingest_directory()
