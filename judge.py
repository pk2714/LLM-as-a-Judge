# ============================================================
#  judge.py  -  STEP 2: ask an LLM to score every response.
#  For each (question, response) the judge LLM gives a 1-5 score
#  for helpfulness, correctness and tone, plus a one-line reason.
#  Output: results/judge_helpsteer.csv and results/judge_custom.csv
#  Run:    python judge.py              (HelpSteer2, plus our set if rated)
#          python judge.py helpsteer    (only one dataset)
#  If it stops (internet, rate limit), just run it again: it continues
#  from where it stopped.
# ============================================================
import csv
import json
import os
import re
import sys
import time

from openai import OpenAI, RateLimitError

import config

# ---------- The judge prompt (the "rubric") ----------
JUDGE_PROMPT = """You are a strict but fair evaluator of AI assistant answers.
Read the user's question and the assistant's response, then rate the response
on three criteria using whole numbers from 1 (worst) to 5 (best).

HELPFULNESS - does it actually solve the user's need?
  1 = useless or ignores the question, 3 = partly useful or vague, 5 = complete and practical
CORRECTNESS - are the facts and instructions true, with nothing important missing?
  1 = mainly wrong, 3 = partly correct or has a mistake, 5 = fully correct
TONE - is it polite, respectful and suitable for the situation?
  1 = rude or insulting, 3 = neutral or a bit cold, 5 = polite, warm and appropriate

Judge each criterion separately (a rude answer can still be correct).
Do not reward length by itself.

QUESTION:
{question}

RESPONSE:
{response}

Reply with ONLY this JSON and nothing else:
{{"helpfulness": <1-5>, "correctness": <1-5>, "tone": <1-5>, "reason": "<one short sentence>"}}"""


def get_api_key():
    """Read the key from the environment variable, or else from api_key.txt."""
    key = os.environ.get(config.API_KEY_ENV, "").strip()
    if not key and os.path.exists(config.API_KEY_FILE):
        key = open(config.API_KEY_FILE, encoding="utf-8").read().strip()
    if "localhost" in config.BASE_URL:
        return key or "not-needed"          # Ollama needs no key
    if not key or "your_key" in key:
        raise SystemExit("No API key found. Paste your Groq key (starts with gsk_) into "
                         "api_key.txt in this folder and run again.")
    return key


def parse_scores(text):
    """Pull the JSON out of the judge's reply and check the scores are 1-5."""
    match = re.search(r"\{.*\}", text, re.DOTALL)   # find the {...} part
    data = json.loads(match.group(0))
    scores = {}
    for c in config.CRITERIA:
        scores[c] = min(5, max(1, int(round(float(data[c])))))
    scores["reason"] = str(data.get("reason", "")).replace("\n", " ")
    return scores


def judge_one(client, question, response, tries=5):
    """Send one question+response to the judge LLM and return its scores."""
    prompt = JUDGE_PROMPT.format(question=question, response=response)
    for attempt in range(tries):
        try:
            extra = {}
            if "gpt-oss" in config.MODEL:            # reasoning models: think briefly, hide the thinking
                extra = {"reasoning_effort": config.REASONING_EFFORT, "include_reasoning": False}
            reply = client.chat.completions.create(
                model=config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.TEMPERATURE,
                extra_body=extra,
            )
            return parse_scores(reply.choices[0].message.content)
        except RateLimitError as error:
            if "per day" in str(error):      # daily free quota is used up
                raise SystemExit("\nDaily free limit of this API key is used up.\n"
                                 "Put ANOTHER team member's Groq key in api_key.txt and run "
                                 "again - it continues from where it stopped.")
            print("   rate limit, waiting 30 s ...")
            time.sleep(30)
        except Exception as error:               # bad JSON, network error ...
            print(f"   attempt {attempt + 1} failed: {str(error)[:150]}")
            if "401" in str(error) or "invalid_api_key" in str(error).lower():
                raise SystemExit("The API key is wrong. Check api_key.txt.")
            if "404" in str(error) or "does not exist" in str(error):
                raise SystemExit(f"The model '{config.MODEL}' is not available to your account. "
                                 "Change MODEL in config.py (e.g. openai/gpt-oss-20b).")
            time.sleep(5)
    raise RuntimeError("Judge failed 5 times on this sample. Run again later.")


def run(client, exp):
    settings = config.EXPERIMENTS[exp]
    if not os.path.exists(settings["samples"]):
        print(f"[{exp}] {settings['samples']} not found - run  python prepare_data.py  first.")
        return
    samples = json.load(open(settings["samples"], encoding="utf-8"))
    out_file = f"results/judge_{exp}.csv"

    # Resume support: skip samples that were already judged in an earlier run.
    done = set()
    if os.path.exists(out_file):
        done = {int(r["id"]) for r in csv.DictReader(open(out_file, encoding="utf-8"))}
    print(f"\n[{exp}] {len(samples)} samples, {len(done)} already done")

    new_file = not os.path.exists(out_file)
    with open(out_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id"] + config.CRITERIA + ["reason"])
        if new_file:
            writer.writeheader()
        for s in samples:
            if s["id"] in done:
                continue
            scores = judge_one(client, s["question"], s["response"])
            writer.writerow({"id": s["id"], **scores})
            f.flush()
            print(f"[{exp}] sample {s['id']:>3}: H={scores['helpfulness']} "
                  f"C={scores['correctness']} T={scores['tone']}  {scores['reason'][:70]}")
            time.sleep(config.SLEEP_SECONDS)
    print(f"[{exp}] done -> {out_file}")


def main():
    os.makedirs("results", exist_ok=True)
    client = OpenAI(base_url=config.BASE_URL, api_key=get_api_key())
    # Default: always HelpSteer2; our own 30 answers only if the team has rated them.
    custom_human = config.EXPERIMENTS["custom"]["human"]
    rated = os.path.exists(custom_human) and len(open(custom_human).read().splitlines()) > 1
    experiments = sys.argv[1:] or (["helpsteer", "custom"] if rated else ["helpsteer"])
    for exp in experiments:
        run(client, exp)


if __name__ == "__main__":
    main()
