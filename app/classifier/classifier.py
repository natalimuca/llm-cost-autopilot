"""Load the trained classifier and score new prompts into complexity tiers."""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from app.classifier.features import FEATURE_NAMES, extract_features

MODEL_PATH = Path(__file__).parent / "model.joblib"


@lru_cache(maxsize=1)
def _model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run `python -m app.classifier.train` first."
        )
    return joblib.load(MODEL_PATH)


def classify(prompt: str) -> tuple[int, float]:
    """Returns (tier, confidence) where tier is 1 (simple), 2 (moderate), or 3 (complex)."""
    features = extract_features(prompt)
    row = pd.DataFrame([features])[FEATURE_NAMES]
    model = _model()
    tier = int(model.predict(row)[0])
    confidence = float(model.predict_proba(row).max())
    return tier, confidence
