from pathlib import Path
import fitz


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents(directory: Path) -> list[dict]:
    """Return document records with text and source metadata."""
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.suffix.lower() == ".pdf":
            with fitz.open(path) as pdf:
                for page_index, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()
                    if text:
                        records.append({
                            "text": text,
                            "source": path.name,
                            "page": page_index,
                            "source_path": str(path),
                        })
        else:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                records.append({
                    "text": text,
                    "source": path.name,
                    "page": None,
                    "source_path": str(path),
                })
    return records


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows, preferring paragraph/sentence boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("; ", start, end),
                           text.rfind(" ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
