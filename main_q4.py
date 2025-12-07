import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam

from utils.preprocessing import load_urdu_english_parallel_corpus
import pickle, os

SAVE_DIR = "saved_models/mt_models/"
GLOVE_PATH = "data/glove.6B.100d.txt"

# ---------------------------
# Load Data
# ---------------------------
en_sentences, ur_sentences = load_urdu_english_parallel_corpus(
    "data/urdu_corpus.txt",
    "data/english_corpus.txt"
)


print(f"[INFO] Loaded {len(en_sentences)} parallel samples.")

# ---------------------------
# Tokenizers
# ---------------------------
from tensorflow.keras.preprocessing.text import Tokenizer

en_tokenizer = Tokenizer(filters='')
ur_tokenizer = Tokenizer(filters='')

en_tokenizer.fit_on_texts(en_sentences)
ur_tokenizer.fit_on_texts(ur_sentences)

with open(SAVE_DIR + "tokenizer_en.pkl", "wb") as f:
    pickle.dump(en_tokenizer, f)

with open(SAVE_DIR + "tokenizer_ur.pkl", "wb") as f:
    pickle.dump(ur_tokenizer, f)

print("[INFO] Tokenizers saved.")

# ---------------------------
# Convert to sequences
# ---------------------------
en_seq = en_tokenizer.texts_to_sequences(en_sentences)
ur_seq = ur_tokenizer.texts_to_sequences(ur_sentences)

max_len_en = max(len(x) for x in en_seq)
max_len_ur = max(len(x) for x in ur_seq)

en_seq = pad_sequences(en_seq, maxlen=max_len_en, padding='post')
ur_seq = pad_sequences(ur_seq, maxlen=max_len_ur, padding='post')

vocab_en = len(en_tokenizer.word_index) + 1
vocab_ur = len(ur_tokenizer.word_index) + 1

# ---------------------------
# Load GloVe embeddings
# ---------------------------
def load_glove():
    embeddings_index = {}
    with open(GLOVE_PATH, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector
    return embeddings_index

print("[INFO] Loading GloVe...")
embeddings_index = load_glove()

# Build embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((vocab_en, embedding_dim))

for word, i in en_tokenizer.word_index.items():
    vec = embeddings_index.get(word)
    if vec is not None:
        embedding_matrix[i] = vec

print("[INFO] GloVe embedding matrix created.")

# ---------------------------
# Build RNN Seq2Seq model
# ---------------------------
def build_seq2seq(embedding_matrix=None, use_glove=False):
    encoder_inputs = Input(shape=(max_len_en,))

    if use_glove:
        encoder_embedding = Embedding(vocab_en, embedding_dim,
                                      weights=[embedding_matrix],
                                      trainable=False)(encoder_inputs)
    else:
        encoder_embedding = Embedding(vocab_en, 128)(encoder_inputs)

    encoder_rnn = SimpleRNN(256, return_state=True)
    _, state = encoder_rnn(encoder_embedding)

    decoder_inputs = Input(shape=(max_len_ur,))
    decoder_embedding = Embedding(vocab_ur, 128)(decoder_inputs)

    decoder_rnn = SimpleRNN(256, return_sequences=True)
    decoder_output = decoder_rnn(decoder_embedding, initial_state=[state])

    outputs = Dense(vocab_ur, activation="softmax")(decoder_output)

    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ---------------------------
# Train Model 1: Random embeddings
# ---------------------------
print("\n[TRAIN] RNN with RANDOM embeddings")
model_random = build_seq2seq(use_glove=False)
model_random.fit([en_seq, ur_seq], ur_seq, epochs=3, batch_size=32)
model_random.save("saved_models/rnn_random_embeddings.keras")

# ---------------------------
# Train Model 2: GloVe embeddings
# ---------------------------
print("\n[TRAIN] RNN with GloVe embeddings")
model_glove = build_seq2seq(embedding_matrix, use_glove=True)
model_glove.fit([en_seq, ur_seq], ur_seq, epochs=3, batch_size=32)
model_glove.save("saved_models/rnn_glove_embeddings.keras")

print("\n[INFO] Both models saved successfully.")
