from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

def build_bilstm_with_embedding(vocab_size, embedding_dim, max_len, embedding_matrix=None, trainable=False):
    input_ = Input(shape=(max_len,))
    if embedding_matrix is not None:
        x = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_len,
            trainable=trainable
        )(input_)
    else:
        x = Embedding(vocab_size, embedding_dim, input_length=max_len)(input_)
    x = LSTM(128, return_sequences=False)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_, outputs=output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
