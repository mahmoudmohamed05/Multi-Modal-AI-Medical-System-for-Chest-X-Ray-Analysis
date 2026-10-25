def format_prediction_context(prediction: dict | None, clinical_context: str = "") -> str:
    """Merge the image-classifier output into the free-text clinical context for the LLM."""
    parts = [clinical_context.strip()] if clinical_context and clinical_context.strip() else []
    if prediction:
        parts.append(
            "Automated chest X-ray classifier output (a screening score from a research model, NOT a "
            f"diagnosis and not verified by a radiologist): label={prediction['label']}, "
            f"P(pneumonia)={prediction['probability_pneumonia']:.3f}, "
            f"decision threshold={prediction['threshold']}, model={prediction['model_name']}."
        )
    return "\n".join(parts)
