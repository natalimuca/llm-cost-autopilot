"""Train the complexity classifier on app/classifier/data/labeled_prompts.csv.

TF-IDF over the raw prompt text, fed into logistic regression. Earlier
versions tried hand-built regex/keyword features (features.py) alone
(~62% held-out accuracy on real data) and TF-IDF combined with those
features (~60%, actively worse -- mismatched feature scales confuse a
linear model fit on a few hundred examples). Text-only TF-IDF is simpler
and outperforms both, plateauing in the mid-60s% no matter how much
regularization or how much more real data is added -- that's the honest
ceiling of a keyword/vocabulary classifier on a 3-way distinction this
fuzzy; getting meaningfully higher would need an LLM-based classifier
(zero/few-shot with a real model) instead of hand-built features.

Re-run this after Phase 3's feedback loop appends new rows from
verifier-caught routing failures.

Usage: python -m app.classifier.train
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "data" / "labeled_prompts.csv"
MODEL_PATH = Path(__file__).parent / "model.joblib"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    prompts, y = df["prompt"], df["tier"]
    prompts_train, prompts_test, y_train, y_test = train_test_split(
        prompts, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1500, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(prompts_train)
    X_test = vectorizer.transform(prompts_test)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=5.0)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Held-out accuracy: {accuracy:.3f}")
    print("Confusion matrix (rows=true, cols=pred, labels=[1,2,3]):")
    print(confusion_matrix(y_test, preds, labels=[1, 2, 3]))
    print(classification_report(y_test, preds))

    joblib.dump({"vectorizer": vectorizer, "clf": clf}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
