"""The flywheel: pull verifier-flagged routing failures out of the audit DB,
append them to the labeled dataset, and retrain the classifier.

Intended to run on a schedule (e.g. weekly cron / Task Scheduler) so the
router keeps improving as it sees more traffic.

Usage: python -m scripts.retrain_from_feedback
"""
import csv

from app.classifier import train
from app.classifier.train import DATA_PATH
from app.db.database import get_unfed_routing_failures, mark_fed_to_training


def main() -> None:
    failures = get_unfed_routing_failures()
    if not failures:
        print("No new routing failures to learn from.")
        return

    with DATA_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in failures:
            writer.writerow([row["prompt"], row["correct_tier"]])
    mark_fed_to_training([row["id"] for row in failures])

    print(f"Appended {len(failures)} routing-failure examples to {DATA_PATH}")
    train.main()


if __name__ == "__main__":
    main()
