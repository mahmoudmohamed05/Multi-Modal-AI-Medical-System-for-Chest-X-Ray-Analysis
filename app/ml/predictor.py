import io
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.ml.model import CLASS_NAMES, IMG_SIZE, MODEL_NAME

DISCLAIMER = (
    "Research prototype. This is an automated image-classification score, not a diagnosis. "
    "A qualified radiologist/clinician must review the image and make all clinical decisions."
)


class InvalidImageError(ValueError):
    """The uploaded bytes are not a usable image."""


class ModelNotAvailableError(RuntimeError):
    """Trained weights are missing or the ML runtime (TensorFlow) is not installed."""


class PneumoniaPredictor:
    """Wraps the notebook's DenseNet121 + cross-attention model.

    A pre-built `model` (anything with `.predict(batch) -> array[(n,1)]`) can be injected for tests.
    """

    def __init__(self, weights_path: Path, threshold: float = 0.5, max_pixels: int = 50_000_000, model=None):
        self.weights_path = Path(weights_path)
        self.threshold = threshold
        self.max_pixels = max_pixels
        self._model = model
        self._lock = threading.Lock()

    # ---- lifecycle ----
    @property
    def weights_present(self) -> bool:
        return self._model is not None or self.weights_path.is_file()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if not self.weights_path.is_file():
                raise ModelNotAvailableError(
                    f"Model weights not found at {self.weights_path}. Copy "
                    "'densenet121_cross_attention.weights.h5' from the notebook's saved_models/ folder, "
                    "or set MODEL_WEIGHTS_PATH in .env."
                )
            try:
                from app.ml.model import build_model
                model = build_model(imagenet_weights=False)
            except ImportError as exc:
                raise ModelNotAvailableError("TensorFlow/Keras is not installed (pip install tensorflow).") from exc
            model.load_weights(str(self.weights_path))
            self._model = model
        return self._model

    # ---- preprocessing (must mirror notebook `load_image`) ----
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """bytes -> float32 array (1, 224, 224, 3) with raw 0-255 pixels.

        Rescaling(1/255) and ImageNet normalisation are inside the model, so none is done here.
        """
        if not image_bytes:
            raise InvalidImageError("Empty file.")
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                if img.width * img.height > self.max_pixels:
                    raise InvalidImageError("Image dimensions are too large.")
                img = img.convert("RGB")          # grayscale X-rays -> 3 channels, like decode_image(channels=3)
                arr = np.asarray(img, dtype=np.float32)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise InvalidImageError("File is not a valid PNG/JPEG image.") from exc
        arr = self._resize(arr)
        return arr[None, ...].astype(np.float32)

    @staticmethod
    def _resize(arr: np.ndarray) -> np.ndarray:
        h, w = IMG_SIZE
        try:
            import tensorflow as tf       # identical to training: tf.image.resize bilinear, then round
            out = tf.image.resize(arr, (h, w), method="bilinear").numpy()
        except ImportError:
            out = np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32)
        return np.round(out).clip(0, 255)

    # ---- inference ----
    def predict(self, image_bytes: bytes) -> dict:
        batch = self.preprocess(image_bytes)
        model = self.load()
        prob = float(np.asarray(model.predict(batch, verbose=0)).ravel()[0])
        prob = min(max(prob, 0.0), 1.0)
        label = CLASS_NAMES[int(prob >= self.threshold)]
        return {
            "label": label,
            "probability_pneumonia": round(prob, 6),
            "confidence": round(prob if label == "PNEUMONIA" else 1.0 - prob, 6),
            "threshold": self.threshold,
            "model_name": MODEL_NAME,
            "disclaimer": DISCLAIMER,
        }


@lru_cache
def get_predictor() -> PneumoniaPredictor:
    s = get_settings()
    return PneumoniaPredictor(s.model_weights_path, s.decision_threshold, s.max_image_pixels)
