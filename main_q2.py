# main_q2.py
import os
import numpy as np
import pandas as pd
from utils.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from gensim.models import Word2Vec, FastText, KeyedVectors
import pickle

# -----------------------
# Paths
# -----------------------
EMBEDDINGS_DIR = "saved_models/embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

w2v_path = os.path.join(EMBEDDINGS_DIR, "w2v_urdu.model")
ft_path  = os.path.join(EMBEDDINGS_DIR, "ft_urdu.model")
glove_path = os.path.join(EMBEDDINGS_DIR, "glove_urdu.model")
elmo_dir = os.path.join(EMBEDDINGS_DIR, "elmo_urdu")  # For future use

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
# 3. Train / Load Embeddings
# -----------------------
# Word2Vec
if os.path.exists(w2v_path):
    print("Loading Word2Vec embeddings...")
    w2v_model = Word2Vec.load(w2v_path)
else:
    print("Training Word2Vec embeddings...")
    w2v_model = Word2Vec(sentences=[t.split() for t in X_train], vector_size=100, window=5, min_count=1, workers=4, epochs=10)
    w2v_model.save(w2v_path)

# FastText
if os.path.exists(ft_path):
    print("Loading FastText embeddings...")
    ft_model = FastText.load(ft_path)
else:
    print("Training FastText embeddings...")
    ft_model = FastText(sentences=[t.split() for t in X_train], vector_size=100, window=5, min_count=1, workers=4, epochs=10)
    ft_model.save(ft_path)

# GloVe (train from corpus using gensim)
if os.path.exists(glove_path):
    print("Loading GloVe embeddings...")
    glove_model = KeyedVectors.load(glove_path)
else:
    print("Training GloVe embeddings (gensim)...")
    from gensim.scripts.glove2word2vec import glove2word2vec
    from gensim.models import Word2Vec
    # Use Word2Vec as a proxy for small corpus GloVe-like embedding
    glove_model = Word2Vec(sentences=[t.split() for t in X_train], vector_size=100, window=5, min_count=1, workers=4, epochs=10)
    glove_model.wv.save(glove_path)

# -----------------------
# 4. Build embedding matrix
# -----------------------
def create_embedding_matrix(tokenizer, word_vectors, embed_size=100):
    matrix = np.zeros((len(tokenizer.word_index) + 1, embed_size))
    for word, i in tokenizer.word_index.items():
        if word in word_vectors:
            matrix[i] = word_vectors[word]
    return matrix

embedding_matrices = {
    "Word2Vec": create_embedding_matrix(tokenizer, w2v_model.wv),
    "FastText": create_embedding_matrix(tokenizer, ft_model.wv),
    "GloVe": create_embedding_matrix(tokenizer, glove_model.wv)
}

# -----------------------
# 5. Build BiLSTM model
# -----------------------
def build_bilstm(embedding_matrix):
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0],
                        output_dim=embedding_matrix.shape[1],
                        weights=[embedding_matrix],
                        input_length=max_len,
                        trainable=False))
    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# -----------------------
# 6. Train & Evaluate
# -----------------------
results = {}

for name, matrix in embedding_matrices.items():
    print(f"\nTraining BiLSTM with {name} embeddings...")
    model = build_bilstm(matrix)
    model.fit(X_train_seq, y_train, validation_split=0.1, epochs=5, batch_size=32, verbose=1)
    y_pred = (model.predict(X_test_seq) > 0.5).astype(int)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

# -----------------------
# 7. Print Results Table
# -----------------------
print("\nFinal Results:")
df_results = pd.DataFrame(results).T
print(df_results)
