from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout

def build_gru(vocab_size, max_len):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        GRU(64),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
