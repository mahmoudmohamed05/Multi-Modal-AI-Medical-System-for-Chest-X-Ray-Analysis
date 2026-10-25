from app.rag.bm25_store import tokenize


def test_tokenize_lowercases_and_keeps_medical_terms():
    assert tokenize("Pneumonia, X-ray and CT-Scan") == [
        "pneumonia", "x-ray", "and", "ct-scan"
    ]
