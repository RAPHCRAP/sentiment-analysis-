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
