from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout

def build_rnn(vocab_size, max_len):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        SimpleRNN(64, return_sequences=False),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
