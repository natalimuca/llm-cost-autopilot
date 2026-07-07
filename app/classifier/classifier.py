"""Load the trained classifier and score new prompts into complexity tiers."""
from functools import lru_cache
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).parent / "model.joblib"


@lru_cache(maxsize=1)
def _bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run `python -m app.classifier.train` first."
        )
    return joblib.load(MODEL_PATH)


def classify(prompt: str) -> tuple[int, float]:
    """Returns (tier, confidence) where tier is 1 (simple), 2 (moderate), or 3 (complex)."""
    bundle = _bundle()
    X = bundle["vectorizer"].transform([prompt])
    clf = bundle["clf"]
    tier = int(clf.predict(X)[0])
    confidence = float(clf.predict_proba(X).max())
    return tier, confidence
