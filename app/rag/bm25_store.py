import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from app.config import get_settings


def tokenize(text: str) -> list[str]:
    # Simple Unicode-aware tokenization; replace with a clinical tokenizer if needed.
    return re.findall(r"[\w-]+", text.lower(), flags=re.UNICODE)


def cache_path() -> Path:
    return Path("data/bm25_index/corpus.json")


def save_corpus(rows: list[dict]) -> None:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def load_corpus() -> list[dict]:
    path = cache_path()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def bm25_search(query: str, top_k: int) -> list[dict]:
    rows = load_corpus()
    if not rows:
        return []
    tokenized = [tokenize(row["text"]) for row in rows]
    if not any(tokenized):
        return []
    model = BM25Okapi(tokenized)
    scores = model.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    return [
        {**rows[index], "bm25_score": float(score)}
        for index, score in ranked[:top_k] if score > 0
    ]
