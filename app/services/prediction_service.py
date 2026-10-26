
import io
import threading

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.ml.model import (
    build_model,
    CLASS_NAMES,
    IMG_SIZE,
    MODEL_NAME,
)

from app.config import get_settings


class PredictionService:
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()

    def load_model(self):
        if self.model is not None:
            return self.model

        with self.lock:
            if self.model is not None:
                return self.model

            settings = get_settings()
            weights_path = settings.model_weights_path

            if not weights_path.is_file():
                raise FileNotFoundError(
                    f"Model weights not found: {weights_path.resolve()}"
                )

            model = build_model(imagenet_weights=False)
            model.load_weights(str(weights_path))
            self.model = model

        return self.model

    @staticmethod
    def preprocess(image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise ValueError("Uploaded image is empty.")

        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image = image.convert("RGB")
                image = image.resize(
                    (IMG_SIZE[1], IMG_SIZE[0]),
                    Image.Resampling.BILINEAR,
                )
                array = np.asarray(image, dtype=np.float32)

        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Upload a valid PNG or JPEG image.") from exc

        return array[None, ...]

    def predict(self, image_bytes: bytes) -> dict:
        batch = self.preprocess(image_bytes)
        model = self.load_model()

        probability = float(
            np.asarray(model.predict(batch, verbose=0)).reshape(-1)[0]
        )
        probability = min(max(probability, 0.0), 1.0)

        index = int(probability >= 0.5)

        return {
            "label": CLASS_NAMES[index],
            "probability_pneumonia": round(probability, 6),
            "threshold": 0.5,
            "model_name": MODEL_NAME,
            "disclaimer": (
                "Research prototype only. This output is not a diagnosis "
                "and requires clinical review."
            ),
        }


prediction_service = PredictionService()
