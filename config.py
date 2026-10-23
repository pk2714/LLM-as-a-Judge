# ============================================================
#  config.py  -  All settings for the project live here.
#  Edit this file only; the other scripts read from it.
# ============================================================

# ---------- 1. Team details (used to build Team_Details.pdf and Report.pdf) ----------
COURSE = "Natural Language Processing - Course Project"
INSTITUTE = "Indian Institute of Information Technology Guwahati (IIITG)"
PROFESSOR = "Prof. ________________"

TEAM_MEMBERS = [
    # (Name, Roll number, Official email)
    ("Tushar", "2301234", "tushar23b@iiitg.ac.in"),
    ("Priyanshu", "2301162", "priyanshu23b@iiitg.ac.in"),
    ("Shashank", "2301195", "shashank23b@iiitg.ac.in"),
]

PROJECT_TITLE = "LLM-as-a-Judge: An Evaluation Pipeline for Helpfulness, Correctness and Tone"

# ---------- 2. Which LLM acts as the judge ----------
# The code works with ANY "OpenAI-compatible" API.
# Option A (recommended, free): Groq  -> free key at https://console.groq.com/keys
# gpt-oss-120b is OpenAI's open-weight 120-billion-parameter model, free on Groq.
# (If it ever says "model does not exist", try "openai/gpt-oss-20b".)
BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"
REASONING_EFFORT = "low"   # gpt-oss "thinks" before answering; low = fast and saves free tokens
#
# Option B (free, runs on your own laptop): Ollama -> https://ollama.com
#   install Ollama, then run:  ollama pull llama3.2
# BASE_URL = "http://localhost:11434/v1"
# MODEL = "llama3.2"

# ---------- 3. API key ----------
# Put your key in a file called  api_key.txt  in this folder (one line, nothing else),
# OR set the environment variable GROQ_API_KEY.
# Do NOT type the key in this file, and do NOT upload api_key.txt to Google Drive.
API_KEY_ENV = "GROQ_API_KEY"
API_KEY_FILE = "api_key.txt"

TEMPERATURE = 0.0      # 0 = the judge gives the same answer every time (repeatable)
SLEEP_SECONDS = 8.0    # pause between API calls (Groq free tier allows about 8,000 tokens per minute)

# ---------- 4. Datasets ----------
# We run TWO experiments:
#  "helpsteer": 120 answers from NVIDIA's public HelpSteer2 dataset, already rated by
#               trained human annotators for helpfulness and correctness (large and real).
#  "custom"   : our own 30 answers rated by our team members, which is how we test TONE
#               (HelpSteer2 has no tone label).
HELPSTEER_URL = "https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/main/validation.jsonl.gz"
HELPSTEER_RAW = "data/helpsteer2_validation.jsonl.gz"
HELPSTEER_N = 120        # how many HelpSteer2 samples to judge
HELPSTEER_MAX_CHARS = 1500   # keep only short single-turn samples (saves free API tokens)
RANDOM_SEED = 42

EXPERIMENTS = {
    "helpsteer": {
        "name": "HelpSteer2",
        "samples": "data/helpsteer_samples.json",
        "human": "data/helpsteer_human.csv",
        "criteria": ["helpfulness", "correctness"],
    },
    "custom": {
        "name": "Our set",
        "samples": "data/samples.json",
        "human": "data/human_ratings.csv",
        "criteria": ["helpfulness", "correctness", "tone"],
    },
}

CRITERIA = ["helpfulness", "correctness", "tone"]   # what the judge always outputs
