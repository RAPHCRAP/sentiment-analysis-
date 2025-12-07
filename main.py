# main.py
from utils.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from models.rnn import build_rnn
from models.gru import build_gru
from models.lstm import build_lstm
from models.bilstm import build_bilstm
from models.bert import load_mbert, predict_mbert
from models.xlmroberta import load_xlm_roberta, predict_xlm

import torch

# -----------------------
# 1. Load & preprocess data
# -----------------------
df = load_dataset("data/urdu-sentiment-corpus.tsv")
df = preprocess(df)

X_train, X_test, y_train, y_test = train_test_split(
    df["tweet"], df["label"], test_size=0.25, random_state=42
)

# -----------------------
# 2. Tokenizer for RNN-based models
# -----------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

vocab_size = len(tokenizer.word_index) + 1
max_len = 50

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

# -----------------------
# 3. Build Keras models
# -----------------------
models_keras = {
    "RNN": build_rnn(vocab_size, max_len),
    "GRU": build_gru(vocab_size, max_len),
    "LSTM": build_lstm(vocab_size, max_len),
    "BiLSTM": build_bilstm(vocab_size, max_len),
}

# -----------------------
# 4. Train & evaluate Keras models
# -----------------------
results = {}

for name, model in models_keras.items():
    print(f"\nTraining {name} Model...")
    model.fit(
        X_train_seq, y_train,
        validation_split=0.1,
        epochs=3,
        batch_size=32,
        verbose=1
    )
    
    # Predict and convert probabilities to class labels
    y_pred_prob = model.predict(X_test_seq)
    if y_pred_prob.shape[1] == 1:  # sigmoid output
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    else:  # softmax output
        y_pred = y_pred_prob.argmax(axis=1)
    
    # Compute all metrics
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary"),
        "recall": recall_score(y_test, y_pred, average="binary"),
        "f1": f1_score(y_test, y_pred, average="binary")
    }

# -----------------------
# 5. mBERT
# -----------------------
print("\nRunning mBERT...")
tok_mbert, mbert_model = load_mbert(X_train, y_train, X_test, y_test, epochs=3)
mbert_preds = predict_mbert(tok_mbert, mbert_model, X_test)
results["mBERT"] = {
    "accuracy": accuracy_score(y_test, mbert_preds),
    "precision": precision_score(y_test, mbert_preds, average="binary"),
    "recall": recall_score(y_test, mbert_preds, average="binary"),
    "f1": f1_score(y_test, mbert_preds, average="binary")
}

# -----------------------
# 6. XLM-RoBERTa
# -----------------------
print("\nRunning XLM-RoBERTa...")
tok_xlm, xlm_model = load_xlm_roberta(X_train, y_train, X_test, y_test, epochs=3)
xlm_preds = predict_xlm(tok_xlm, xlm_model, X_test)
results["XLM-RoBERTa"] = {
    "accuracy": accuracy_score(y_test, xlm_preds),
    "precision": precision_score(y_test, xlm_preds, average="binary"),
    "recall": recall_score(y_test, xlm_preds, average="binary"),
    "f1": f1_score(y_test, xlm_preds, average="binary")
}

# -----------------------
# 7. Print final results
# -----------------------
print("\nFinal Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
