# ============================================================
#  rate.py  -  STEP 1: collect HUMAN ratings.
#  (Only for OUR 30-answer set; HelpSteer2 already has human ratings.)
#  Each team member runs this once. It shows every response and
#  asks for three scores (1-5). Answers go into data/human_ratings.csv
#  Run:    python rate.py
#  (You can also type ratings straight into the CSV in Excel.)
# ============================================================
import csv
import json
import os

import config

SAMPLES_FILE = config.EXPERIMENTS["custom"]["samples"]
HUMAN_FILE = config.EXPERIMENTS["custom"]["human"]

GUIDE = """
Score each response from 1 (worst) to 5 (best):
  helpfulness : 1 useless/ignores question, 3 partly useful, 5 complete and practical
  correctness : 1 mainly wrong, 3 partly correct, 5 fully correct
  tone        : 1 rude/insulting, 3 neutral or cold, 5 polite, warm and suitable
Rate the three things separately. Do NOT look at the judge's scores first.
"""


def ask(name):
    """Keep asking until the person types a number from 1 to 5."""
    while True:
        value = input(f"   {name} (1-5): ").strip()
        if value in {"1", "2", "3", "4", "5"}:
            return int(value)
        print("   Please type a whole number from 1 to 5.")


def main():
    samples = json.load(open(SAMPLES_FILE, encoding="utf-8"))
    rater = input("Your name (e.g. rater1 or your first name): ").strip() or "rater"
    print(GUIDE)

    # Find what this rater has already done, so they can stop and continue later.
    done = set()
    if os.path.exists(HUMAN_FILE):
        for r in csv.DictReader(open(HUMAN_FILE, encoding="utf-8")):
            if r["rater"] == rater:
                done.add(int(r["id"]))

    new_file = not os.path.exists(HUMAN_FILE) or os.path.getsize(HUMAN_FILE) == 0
    with open(HUMAN_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rater", "id"] + config.CRITERIA)
        if new_file:
            writer.writeheader()
        for s in samples:
            if s["id"] in done:
                continue
            print("=" * 70)
            print(f"Sample {s['id']} of {len(samples)}")
            print(f"QUESTION: {s['question']}")
            print(f"RESPONSE: {s['response']}")
            row = {"rater": rater, "id": s["id"]}
            for c in config.CRITERIA:
                row[c] = ask(c)
            writer.writerow(row)
            f.flush()

    print(f"\nThank you! Ratings saved to {HUMAN_FILE}")


if __name__ == "__main__":
    main()
