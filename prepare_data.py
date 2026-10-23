# ============================================================
#  prepare_data.py  -  STEP 0: download and prepare HelpSteer2.
#  HelpSteer2 (NVIDIA, 2024) is a public dataset of prompts and model
#  responses, each rated 0-4 by human annotators for helpfulness,
#  correctness, coherence, complexity and verbosity.
#  We take a sample of short, single-turn examples and convert the
#  human scores from 0-4 to our 1-5 scale (add 1).
#  Output: data/helpsteer_samples.json, data/helpsteer_human.csv
#  Run:    python prepare_data.py
# ============================================================
import csv
import gzip
import json
import os
import random
import urllib.request

import config


def download():
    """Download the HelpSteer2 validation split (about 0.6 MB) if we don't have it yet."""
    if os.path.exists(config.HELPSTEER_RAW):
        return
    print("Downloading HelpSteer2 validation split ...")
    request = urllib.request.Request(config.HELPSTEER_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as r, open(config.HELPSTEER_RAW, "wb") as f:
            f.write(r.read())
    except Exception as error:
        raise SystemExit(
            f"Download failed ({error}).\nOpen this link in your browser, save the file into the "
            f"data folder as 'helpsteer2_validation.jsonl.gz', then run this script again:\n"
            f"{config.HELPSTEER_URL}")


def main():
    os.makedirs("data", exist_ok=True)
    download()

    rows = [json.loads(line) for line in gzip.open(config.HELPSTEER_RAW, "rt", encoding="utf-8")]
    print(f"HelpSteer2 validation split: {len(rows)} samples")

    # Keep single-turn (no earlier conversation) and short samples.
    keep = [r for r in rows
            if "<extra_id_1>" not in r["prompt"]
            and r["response"].strip()
            and len(r["prompt"]) + len(r["response"]) <= config.HELPSTEER_MAX_CHARS]
    print(f"Short single-turn samples: {len(keep)}")

    # Stratified sample: try to take the same number from each helpfulness level (0-4),
    # so the judge is tested on bad, average AND good answers (not mostly good ones).
    random.seed(config.RANDOM_SEED)
    per_level = config.HELPSTEER_N // 5
    chosen = []
    for level in range(5):
        group = [r for r in keep if r["helpfulness"] == level]
        chosen += random.sample(group, min(per_level, len(group)))
    rest = [r for r in keep if r not in chosen]
    chosen += random.sample(rest, config.HELPSTEER_N - len(chosen))   # fill up if a level was short
    random.shuffle(chosen)

    samples, human = [], []
    for i, r in enumerate(chosen, 1):
        samples.append({"id": i, "question": r["prompt"].strip(), "response": r["response"].strip()})
        human.append({"rater": "helpsteer2_annotators", "id": i,
                      "helpfulness": r["helpfulness"] + 1,      # 0-4  ->  1-5
                      "correctness": r["correctness"] + 1})

    json.dump(samples, open(config.EXPERIMENTS["helpsteer"]["samples"], "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    with open(config.EXPERIMENTS["helpsteer"]["human"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rater", "id", "helpfulness", "correctness"])
        writer.writeheader()
        writer.writerows(human)

    counts = {lvl: sum(h["helpfulness"] == lvl for h in human) for lvl in range(1, 6)}
    print(f"Saved {len(samples)} samples. Human helpfulness scores (1-5) -> count: {counts}")


if __name__ == "__main__":
    main()
