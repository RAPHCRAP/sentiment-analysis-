from utils.preprocessing import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models.rnn import build_rnn
from models.gru import build_gru
from models.lstm import build_lstm
from models.bilstm import build_bilstm

# -----------------------
# 1. Load & preprocess data
# -----------------------

df = load_dataset("data/urdu-sentiment-corpus.tsv")
df = preprocess(df)

X_train, X_test, y_train, y_test = train_test_split(
    df["tweet"], df["label"], test_size=0.25, random_state=42
)

# -----------------------
# 2. Tokenizer
# -----------------------

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

vocab_size = len(tokenizer.word_index) + 1
max_len = 50

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

# -----------------------
# 3. Build models
# -----------------------

models = {
    "RNN": build_rnn(vocab_size, max_len),
    "GRU": build_gru(vocab_size, max_len),
    "LSTM": build_lstm(vocab_size, max_len),
    "BiLSTM": build_bilstm(vocab_size, max_len),
}

# -----------------------
# 4. Train & evaluate
# -----------------------

results = {}

for name, model in models.items():
    print(f"\nTraining {name} Model...")
    model.fit(
        X_train_seq, y_train,
        validation_split=0.1,
        epochs=3,
        batch_size=32,
        verbose=1
    )

    loss, acc = model.evaluate(X_test_seq, y_test)
    results[name] = acc

print("\nFinal Results:")
print(results)
