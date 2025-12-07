# main_q3.py
"""
Q3: Comparative Seq2Seq Study (English → Urdu)

This updated version loads:
    data/english_corpus.txt
    data/urdu_corpus.txt

Make sure both files have SAME NUMBER OF LINES.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
nltk.download("punkt")
nltk.download('punkt_tab')


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models.seq2seq_models import (
    build_rnn_seq2seq,
    build_birnn_seq2seq,
    build_lstm_seq2seq,
    build_bilstm_seq2seq,
    build_transformer_seq2seq,
    greedy_decode
)

# ================================
# CONFIG
# ================================
EN_FILE = "data/english_corpus.txt"
UR_FILE = "data/urdu_corpus.txt"

OUTPUT_DIR = "saved_models/mt_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOKENIZER_DIR = OUTPUT_DIR + "/tokenizers"
os.makedirs(TOKENIZER_DIR, exist_ok=True)

EMBED_DIM = 256
LATENT_DIM = 512
EPOCHS = 50
BATCH_SIZE = 64
EN_MAXLEN = 30
UR_MAXLEN = 40
NUM_HEADS = 4
FF_DIM = 512
NUM_LAYERS = 2

START_TOKEN = "<sos>"
END_TOKEN = "<eos>"

# ================================
# 1. LOAD TXT PARALLEL DATA
# ================================
print("Loading TXT dataset...")

with open(EN_FILE, "r", encoding="utf-8") as f:
    en_texts = [line.strip() for line in f.readlines()]

with open(UR_FILE, "r", encoding="utf-8") as f:
    ur_texts = [line.strip() for line in f.readlines()]

assert len(en_texts) == len(ur_texts), "❌ English and Urdu files do NOT have equal lines!"

# Light token normalization
def normalize_en(s):
    return " ".join(nltk.word_tokenize(s.lower()))

en_texts = [normalize_en(t) for t in en_texts]
ur_texts = [t for t in ur_texts]

# Add start/end tokens to Urdu target
ur_texts_in = [f"{START_TOKEN} {t} {END_TOKEN}" for t in ur_texts]


# ================================
# 2. TOKENIZERS (Load or Train)
# ================================
tok_en_path = TOKENIZER_DIR + "/tokenizer_en.pkl"
tok_ur_path = TOKENIZER_DIR + "/tokenizer_ur.pkl"

if os.path.exists(tok_en_path) and os.path.exists(tok_ur_path):
    print("Loading saved tokenizers...")
    tokenizer_en = pickle.load(open(tok_en_path, "rb"))
    tokenizer_ur = pickle.load(open(tok_ur_path, "rb"))
else:
    print("Training new tokenizers...")
    tokenizer_en = Tokenizer(oov_token="<unk>")
    tokenizer_ur = Tokenizer(oov_token="<unk>")

    tokenizer_en.fit_on_texts(en_texts)
    tokenizer_ur.fit_on_texts(ur_texts_in)

    pickle.dump(tokenizer_en, open(tok_en_path, "wb"))
    pickle.dump(tokenizer_ur, open(tok_ur_path, "wb"))

en_vocab = len(tokenizer_en.word_index) + 1
ur_vocab = len(tokenizer_ur.word_index) + 1
print("English Vocab:", en_vocab, "Urdu Vocab:", ur_vocab)


# ================================
# 3. TEXT → SEQUENCES
# ================================
en_seq = pad_sequences(tokenizer_en.texts_to_sequences(en_texts), maxlen=EN_MAXLEN, padding="post")
ur_seq = pad_sequences(tokenizer_ur.texts_to_sequences(ur_texts_in), maxlen=UR_MAXLEN, padding="post")

# Decoder input/target shift
dec_input = pad_sequences([seq[:-1] for seq in ur_seq], maxlen=UR_MAXLEN, padding="post")
dec_target = pad_sequences([seq[1:] for seq in ur_seq], maxlen=UR_MAXLEN, padding="post")
dec_target = np.expand_dims(dec_target, -1)


# ================================
# 4. TRAIN-TEST SPLIT
# ================================
(
    X_en_train, X_en_test,
    X_dec_in_train, X_dec_in_test,
    y_train, y_test
) = train_test_split(
    en_seq, dec_input, dec_target, test_size=0.25, random_state=42
)

_, en_test_raw, _, ur_test_raw = train_test_split(
    en_texts, ur_texts, test_size=0.25, random_state=42
)


# ================================
# 5. BUILD MODEL DICTIONARY
# ================================
models_to_run = {
    "RNN": build_rnn_seq2seq(en_vocab, ur_vocab, EN_MAXLEN, UR_MAXLEN, EMBED_DIM, LATENT_DIM),
    "BiRNN": build_birnn_seq2seq(en_vocab, ur_vocab, EN_MAXLEN, UR_MAXLEN, EMBED_DIM, LATENT_DIM),
    "LSTM": build_lstm_seq2seq(en_vocab, ur_vocab, EN_MAXLEN, UR_MAXLEN, EMBED_DIM, LATENT_DIM),
    "BiLSTM": build_bilstm_seq2seq(en_vocab, ur_vocab, EN_MAXLEN, UR_MAXLEN, EMBED_DIM, LATENT_DIM),
    "Transformer": build_transformer_seq2seq(
        en_vocab, ur_vocab, EN_MAXLEN, UR_MAXLEN,
        EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS
    ),
}

results = {}

start_id = tokenizer_ur.word_index.get(START_TOKEN)
end_id = tokenizer_ur.word_index.get(END_TOKEN)
inv_ur = {v: k for k, v in tokenizer_ur.word_index.items()}


# ================================
# 6. TRAIN / LOAD AND EVALUATE
# ================================
for name, model in models_to_run.items():
    print("\n===============================")
    print(f"Running Model: {name}")
    print("===============================")

    model_path = f"{OUTPUT_DIR}/{name.lower()}_mt.keras"

    if os.path.exists(model_path):
        print(f"Loading saved {name} model...")
        model = tf.keras.models.load_model(model_path, compile=True)
    else:
        print(f"Training {name}...")
        model.fit(
            [X_en_train, X_dec_in_train], y_train,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        model.save(model_path)

    # Decode first 64 test sentences
    decoded_id_seqs = greedy_decode(
        model, X_en_test[:64], start_id, end_id, UR_MAXLEN,
        model_type=("transformer" if name == "Transformer" else "rnn")
    )

    # Convert ids → text
    hypotheses = []
    for seq in decoded_id_seqs:
        words = []
        for t in seq:
            if t == 0: continue
            w = inv_ur.get(int(t), "<unk>")
            if w == END_TOKEN: break
            if w != START_TOKEN:
                words.append(w)
        hypotheses.append(words)

    references = [[r.split()] for r in ur_test_raw[:len(hypotheses)]]

    try:
        bleu = corpus_bleu(references, hypotheses)
    except:
        bleu = np.mean([sentence_bleu(references[i], hypotheses[i]) for i in range(len(hypotheses))])

    results[name] = bleu
    print(f"{name} BLEU = {bleu:.4f}")

# ================================
# 7. SAVE RESULTS
# ================================
df_res = pd.DataFrame.from_dict(results, orient="index", columns=["BLEU"])
df_res.to_csv(f"{OUTPUT_DIR}/mt_results_bleu.csv")

print("\n==== FINAL BLEU SCORES ====")
print(df_res)
print("\nSaved results to:", OUTPUT_DIR)
