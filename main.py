# main.py
import os
from utils.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from models.rnn import build_rnn
from models.gru import build_gru
from models.lstm import build_lstm
from models.bilstm import build_bilstm

from models.bert import load_mbert, predict_mbert
from models.xlmroberta import load_xlm_roberta, predict_xlm


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
# 3. Build OR Load RNN/GRU/LSTM/BiLSTM Models
# -----------------------
models_keras = {
    "RNN": ("saved_models/rnn_model.keras", build_rnn(vocab_size, max_len)),
    "GRU": ("saved_models/gru_model.keras", build_gru(vocab_size, max_len)),
    "LSTM": ("saved_models/lstm_model.keras", build_lstm(vocab_size, max_len)),
    "BiLSTM": ("saved_models/bilstm_model.keras", build_bilstm(vocab_size, max_len)),
}

results = {}

for name, (path, model) in models_keras.items():
    if os.path.exists(path):
        print(f"\nLoading saved {name} model...")
        model = load_model(path, compile=True)
    else:
        print(f"\nTraining {name} Model...")
        model.fit(
            X_train_seq, y_train,
            validation_split=0.1,
            epochs=3,
            batch_size=32,
            verbose=1
        )
        model.save(path, save_format="keras")

    # Predictions
    y_pred = (model.predict(X_test_seq) > 0.5).astype(int)

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }


# -----------------------
# 4. mBERT
# -----------------------
mbert_path = "saved_models/mbert"

if os.path.exists(mbert_path) and os.listdir(mbert_path):
    print("\nLoading saved mBERT...")
    tok_mbert, mbert_model = load_mbert(load_saved=True)
else:
    print("\nTraining mBERT...")
    tok_mbert, mbert_model = load_mbert(X_train, y_train, X_test, y_test, epochs=3)
    mbert_model.save_pretrained(mbert_path)
    tok_mbert.save_pretrained(mbert_path)

mbert_preds = predict_mbert(tok_mbert, mbert_model, X_test)
results["mBERT"] = {
    "accuracy": accuracy_score(y_test, mbert_preds),
    "precision": precision_score(y_test, mbert_preds),
    "recall": recall_score(y_test, mbert_preds),
    "f1": f1_score(y_test, mbert_preds)
}


# -----------------------
# 5. XLM-RoBERTa
# -----------------------
xlm_path = "saved_models/xlm"

if os.path.exists(xlm_path) and os.listdir(xlm_path):
    print("\nLoading saved XLM-RoBERTa...")
    tok_xlm, xlm_model = load_xlm_roberta(load_saved=True)
else:
    print("\nTraining XLM-RoBERTa...")
    tok_xlm, xlm_model = load_xlm_roberta(X_train, y_train, X_test, y_test, epochs=3)
    xlm_model.save_pretrained(xlm_path)
    tok_xlm.save_pretrained(xlm_path)

xlm_preds = predict_xlm(tok_xlm, xlm_model, X_test)
results["XLM-RoBERTa"] = {
    "accuracy": accuracy_score(y_test, xlm_preds),
    "precision": precision_score(y_test, xlm_preds),
    "recall": recall_score(y_test, xlm_preds),
    "f1": f1_score(y_test, xlm_preds)
}


# -----------------------
# 6. Final Results
# -----------------------
print("\nFinal Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
