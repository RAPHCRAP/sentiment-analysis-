import pandas as pd
import re

def load_dataset(path):
    return pd.read_csv(path, sep="\t")


def clean_text(text):
    text = str(text)
    text = re.sub(r"[!؟?,،.…\"\']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess(df):
    # Rename your columns to match what main.py expects
    df = df.rename(columns={
        "Tweet": "tweet",
        "Class": "label"
    })

    # Remove neutral / "O" class if present
    df = df[df["label"].isin(["P", "N"])]

    df["tweet"] = df["tweet"].astype(str).apply(clean_text)

    # Encode labels: P → 1, N → 0
    df["label"] = df["label"].map({"P": 1, "N": 0})

    return df


def encode_transformer(tokenizer, texts, max_len=64):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="tf"
    )


def load_urdu_english_parallel_corpus(urdu_path, english_path, limit=None):
    """
    Loads a parallel Urdu-English corpus from two aligned text files.
    Each line in urdu_path must correspond to the same line number in english_path.
    """
    urdu_lines = []
    english_lines = []

    with open(urdu_path, "r", encoding="utf-8") as f_ur:
        urdu_lines = f_ur.read().strip().split("\n")

    with open(english_path, "r", encoding="utf-8") as f_en:
        english_lines = f_en.read().strip().split("\n")

    # Safety check: ensure equal sizes
    min_len = min(len(urdu_lines), len(english_lines))
    urdu_lines = urdu_lines[:min_len]
    english_lines = english_lines[:min_len]

    if limit:
        urdu_lines = urdu_lines[:limit]
        english_lines = english_lines[:limit]

    print(f"[INFO] Loaded {len(urdu_lines)} parallel sentences.")

    return urdu_lines, english_lines
