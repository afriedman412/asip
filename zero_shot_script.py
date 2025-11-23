from datasets import Dataset
import pandas as pd
from transformers import pipeline
import os
import sys
from datetime import datetime

# --- choose where you want to save outputs ---

# Option 1: same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Option 2: subdirectory for outputs
SAVE_DIR = os.path.join(BASE_DIR, "zs_output")
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"[{datetime.now()}] Using SAVE_DIR={SAVE_DIR}", flush=True)

# --- quick write test to make sure we have permissions ---

test_path = os.path.join(SAVE_DIR, "_write_test.tmp")

try:
    with open(test_path, "w") as f:
        f.write("test")
    os.remove(test_path)
    print(f"[{datetime.now()}] Write test OK in {SAVE_DIR}", flush=True)
except Exception as e:
    print(f"[{datetime.now()}] ERROR: Cannot write to {SAVE_DIR}: {e}", flush=True)
    sys.exit(1)  # bail out before doing the expensive zero-shot run

# THE RUN STARTS HERE

df = pd.read_csv('data/all_shows_chunked.csv')
df.rename(columns={"TextConcat": "text"}, inplace=True)
chunk_ds = Dataset.from_pandas(
    df[['show', 'Season', 'Episode', 'chunk_idx', 'text']].dropna()
)

TOTAL_ROWS = len(chunk_ds)
print(f"[{datetime.now()}] Starting zero-shot on {TOTAL_ROWS} rows", flush=True)

ZS_LABELS = [
    "This scene is about socioeconomic class and inequality.",
    "This scene is about race and ethnicity and racial or ethnic identity.",
    "This scene is about electoral politics and political ideology.",
    "This scene is about religion and faith and associated identity issues.",
    "This scene is about LGBTQ issues and queer identity.",
    "This scene is about gun policy and firearms.",
    "This scene is about illegal drugs and substance abuse and policy.",
    "This scene is not about any of the above.",
]

MAPPING = dict(zip(ZS_LABELS, [
    'CLASS', 'RACE', 'POLITICS', 'RELIGION',
    'GAYS', 'GUNS', 'DRUGS', "NONE"
]))

zero_shot = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    truncation=True,
    max_length=512,
    multi_label=True
)

# simple global counters for logging
PROCESSED = 0
LOG_EVERY = 5000  # rows


def run_zero_shot(batch):
    global PROCESSED

    out = zero_shot(
        batch["text"],
        candidate_labels=ZS_LABELS,
        multi_label=True
    )

    # Build a short-code dictionary per text
    mapped = []
    for o in out:
        d = {}
        for label, score in zip(o["labels"], o["scores"]):
            short = MAPPING[label]
            d[short] = score
        mapped.append(d)

    # ---- logging: update counter and occasionally print progress ----
    PROCESSED += len(batch["text"])
    if PROCESSED % LOG_EVERY < len(batch["text"]):
        print(
            f"[{datetime.now()}] Processed {PROCESSED}/{TOTAL_ROWS} rows",
            flush=True
        )

    return {
        "topic_scores": mapped,      # fully mapped dicts
        "zs_raw_labels": [o["labels"] for o in out],   # optional debugging
        "zs_raw_scores": [o["scores"] for o in out],   # optional debugging
    }


chunk_topics = chunk_ds.map(run_zero_shot, batched=True, batch_size=16)

print(f"[{datetime.now()}] Finished zero-shot on {TOTAL_ROWS} rows", flush=True)
print(f"[{datetime.now()}] Saving...", flush=True)

dataset_path = os.path.join(SAVE_DIR, "hf_dataset")
csv_path = os.path.join(SAVE_DIR, "zs_output.csv")

chunk_topics.save_to_disk("zs_output/")
chunk_topics.to_pandas().to_csv("zs_output.csv", index=False)

print(f"[{datetime.now()}] Saved results to zs_output/ and zs_output.csv", flush=True)
