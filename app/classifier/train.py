"""Train the complexity classifier on app/classifier/data/labeled_prompts.csv.

Not optimizing for classifier perfection here — this is the routing skeleton.
>80% held-out accuracy is fine for V1. Re-run this after Phase 3's feedback
loop appends new rows from verifier-caught routing failures.

Usage: python -m app.classifier.train
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from app.classifier.features import FEATURE_NAMES, extract_features

DATA_PATH = Path(__file__).parent / "data" / "labeled_prompts.csv"
MODEL_PATH = Path(__file__).parent / "model.joblib"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    feature_rows = [extract_features(p) for p in df["prompt"]]
    X = pd.DataFrame(feature_rows)[FEATURE_NAMES]
    y = df["tier"]
    return X, y


def main() -> None:
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Held-out accuracy: {accuracy:.3f}")
    print("Confusion matrix (rows=true, cols=pred, labels=[1,2,3]):")
    print(confusion_matrix(y_test, preds, labels=[1, 2, 3]))
    print(classification_report(y_test, preds))

    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
