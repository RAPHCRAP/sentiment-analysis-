# models/seq2seq_models.py
"""
Seq2Seq model definitions and helpers:
- RNN (vanilla) encoder-decoder
- BiRNN encoder + RNN decoder
- LSTM encoder-decoder
- BiLSTM encoder + LSTM decoder
- Transformer encoder-decoder (simple Keras implementation)
Also includes greedy decoding helpers for each model type.

All models use Keras (tf.keras).
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, SimpleRNN, LSTM, GRU, Dense, TimeDistributed,
    Bidirectional, Concatenate, Dropout, LayerNormalization, MultiHeadAttention
)
import numpy as np

# ------------------------------
# Helper: Positional encoding for Transformer
# ------------------------------
def positional_encoding(maxlen, d_model):
    pos = np.arange(maxlen)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    # apply sin to even indices; cos to odd indices
    sines = np.sin(angle_rads[:, 0::2])
    coses = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = coses
    return tf.cast(pos_encoding, dtype=tf.float32)

# ------------------------------
# 1) Vanilla RNN seq2seq
# ------------------------------
def build_rnn_seq2seq(
    en_vocab_size, ur_vocab_size,
    en_maxlen, ur_maxlen,
    embedding_dim=256, latent_dim=512
):
    # encoder
    enc_inputs = Input(shape=(en_maxlen,), name="enc_inputs")
    enc_emb = Embedding(en_vocab_size, embedding_dim, mask_zero=True, name="enc_emb")(enc_inputs)
    enc_rnn = SimpleRNN(latent_dim, return_state=True, name="enc_rnn")
    enc_output, enc_state = enc_rnn(enc_emb)

    # decoder
    dec_inputs = Input(shape=(ur_maxlen,), name="dec_inputs")
    dec_emb = Embedding(ur_vocab_size, embedding_dim, mask_zero=True, name="dec_emb")(dec_inputs)
    dec_rnn = SimpleRNN(latent_dim, return_sequences=True, return_state=True, name="dec_rnn")
    dec_output, _ = dec_rnn(dec_emb, initial_state=enc_state)
    dec_dense = TimeDistributed(Dense(ur_vocab_size, activation="softmax"), name="dec_out")
    dec_outputs = dec_dense(dec_output)

    model = Model([enc_inputs, dec_inputs], dec_outputs, name="rnn_seq2seq")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ------------------------------
# 2) BiRNN encoder + RNN decoder
# ------------------------------
def build_birnn_seq2seq(
    en_vocab_size, ur_vocab_size,
    en_maxlen, ur_maxlen,
    embedding_dim=256, latent_dim=512
):
    enc_inputs = Input(shape=(en_maxlen,), name="enc_inputs")
    enc_emb = Embedding(en_vocab_size, embedding_dim, mask_zero=True, name="enc_emb")(enc_inputs)
    # bidirectional RNN
    enc_bi = Bidirectional(SimpleRNN(latent_dim, return_sequences=False, return_state=False), name="enc_birnn")
    enc_output = enc_bi(enc_emb)  # returns a single vector (concatenated)
    # Map to decoder initial state size
    enc_state = Dense(latent_dim, activation="tanh", name="enc_state_proj")(enc_output)

    dec_inputs = Input(shape=(ur_maxlen,), name="dec_inputs")
    dec_emb = Embedding(ur_vocab_size, embedding_dim, mask_zero=True, name="dec_emb")(dec_inputs)
    dec_rnn = SimpleRNN(latent_dim, return_sequences=True, return_state=True, name="dec_rnn")
    dec_output, _ = dec_rnn(dec_emb, initial_state=enc_state)
    dec_dense = TimeDistributed(Dense(ur_vocab_size, activation="softmax"), name="dec_out")
    dec_outputs = dec_dense(dec_output)

    model = Model([enc_inputs, dec_inputs], dec_outputs, name="birnn_seq2seq")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ------------------------------
# 3) LSTM seq2seq
# ------------------------------
def build_lstm_seq2seq(
    en_vocab_size, ur_vocab_size,
    en_maxlen, ur_maxlen,
    embedding_dim=256, latent_dim=512
):
    enc_inputs = Input(shape=(en_maxlen,), name="enc_inputs")
    enc_emb = Embedding(en_vocab_size, embedding_dim, mask_zero=True, name="enc_emb")(enc_inputs)
    enc_lstm = LSTM(latent_dim, return_state=True, name="enc_lstm")
    _, state_h, state_c = enc_lstm(enc_emb)
    enc_states = [state_h, state_c]

    dec_inputs = Input(shape=(ur_maxlen,), name="dec_inputs")
    dec_emb = Embedding(ur_vocab_size, embedding_dim, mask_zero=True, name="dec_emb")(dec_inputs)
    dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="dec_lstm")
    dec_output, _, _ = dec_lstm(dec_emb, initial_state=enc_states)
    dec_dense = TimeDistributed(Dense(ur_vocab_size, activation="softmax"), name="dec_out")
    dec_outputs = dec_dense(dec_output)

    model = Model([enc_inputs, dec_inputs], dec_outputs, name="lstm_seq2seq")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ------------------------------
# 4) BiLSTM encoder + LSTM decoder
# ------------------------------
def build_bilstm_seq2seq(
    en_vocab_size, ur_vocab_size,
    en_maxlen, ur_maxlen,
    embedding_dim=256, latent_dim=512
):
    enc_inputs = Input(shape=(en_maxlen,), name="enc_inputs")
    enc_emb = Embedding(en_vocab_size, embedding_dim, mask_zero=True, name="enc_emb")(enc_inputs)
    enc_bi = Bidirectional(LSTM(latent_dim, return_sequences=False), name="enc_bilstm")
    enc_out = enc_bi(enc_emb)
    # project concatenated forward+back to decoder latent size
    enc_state_h = Dense(latent_dim, activation="tanh", name="enc_proj_h")(enc_out)
    enc_state_c = Dense(latent_dim, activation="tanh", name="enc_proj_c")(enc_out)
    enc_states = [enc_state_h, enc_state_c]

    dec_inputs = Input(shape=(ur_maxlen,), name="dec_inputs")
    dec_emb = Embedding(ur_vocab_size, embedding_dim, mask_zero=True, name="dec_emb")(dec_inputs)
    dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="dec_lstm")
    dec_output, _, _ = dec_lstm(dec_emb, initial_state=enc_states)
    dec_dense = TimeDistributed(Dense(ur_vocab_size, activation="softmax"), name="dec_out")
    dec_outputs = dec_dense(dec_output)

    model = Model([enc_inputs, dec_inputs], dec_outputs, name="bilstm_seq2seq")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ------------------------------
# 5) Simple Transformer encoder-decoder
# (small educational implementation)
# ------------------------------
def build_transformer_seq2seq(
    en_vocab_size, ur_vocab_size,
    en_maxlen, ur_maxlen,
    embedding_dim=256, num_heads=4, ff_dim=512, num_layers=2
):
    # Encoder
    enc_inputs = Input(shape=(en_maxlen,), name="enc_inputs")
    enc_emb = Embedding(en_vocab_size, embedding_dim, mask_zero=True)(enc_inputs)
    pos_enc = positional_encoding(en_maxlen, embedding_dim)
    enc = enc_emb + pos_enc[tf.newaxis, :en_maxlen, :]

    for i in range(num_layers):
        # Self-attention
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim//num_heads, name=f"enc_mha_{i}")(enc, enc)
        attn = Dropout(0.1)(attn)
        enc = LayerNormalization(epsilon=1e-6)(enc + attn)
        # FFN
        ffn = Dense(ff_dim, activation="relu")(enc)
        ffn = Dense(embedding_dim)(ffn)
        ffn = Dropout(0.1)(ffn)
        enc = LayerNormalization(epsilon=1e-6)(enc + ffn)

    # Decoder
    dec_inputs = Input(shape=(ur_maxlen,), name="dec_inputs")
    dec_emb = Embedding(ur_vocab_size, embedding_dim, mask_zero=True)(dec_inputs)
    dec = dec_emb + positional_encoding(ur_maxlen, embedding_dim)[tf.newaxis, :ur_maxlen, :]

    for i in range(num_layers):
        # Self-attention (look-ahead)
        attn1 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim//num_heads, name=f"dec_mha1_{i}")(dec, dec)
        attn1 = Dropout(0.1)(attn1)
        dec = LayerNormalization(epsilon=1e-6)(dec + attn1)
        # Cross attention: decoder attends to encoder output
        attn2 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim//num_heads, name=f"dec_mha2_{i}")(dec, enc)
        attn2 = Dropout(0.1)(attn2)
        dec = LayerNormalization(epsilon=1e-6)(dec + attn2)
        # FFN
        ffn = Dense(ff_dim, activation="relu")(dec)
        ffn = Dense(embedding_dim)(ffn)
        ffn = Dropout(0.1)(ffn)
        dec = LayerNormalization(epsilon=1e-6)(dec + ffn)

    dec_out = TimeDistributed(Dense(ur_vocab_size, activation="softmax"), name="dec_out")(dec)

    model = Model([enc_inputs, dec_inputs], dec_out, name="transformer_seq2seq")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ------------------------------
# Greedy decoding helpers
# ------------------------------
def greedy_decode_rnn(model, enc_input_seq, start_token, end_token, ur_maxlen):
    """
    Greedy decode for decoder that expects full target input sequence.
    We'll do iterative feeding: at each step, provide decoded tokens so far.
    This function assumes the same model signature [enc_inputs, dec_inputs] -> outputs
    """
    batch_size = enc_input_seq.shape[0]
    decoded = np.zeros((batch_size, ur_maxlen), dtype="int32")
    # initialize first token as start_token
    decoded[:,0] = start_token

    for t in range(1, ur_maxlen):
        preds = model.predict([enc_input_seq, decoded], verbose=0)  # (batch, ur_maxlen, vocab)
        next_token = np.argmax(preds[:, t-1, :], axis=-1)
        decoded[:, t] = next_token
    # convert to list of token ids per sample
    return decoded

def greedy_decode_transformer(model, enc_input_seq, start_token, end_token, ur_maxlen):
    # same iterative approach: feed decoder inputs progressively
    batch_size = enc_input_seq.shape[0]
    decoded = np.zeros((batch_size, ur_maxlen), dtype="int32")
    decoded[:,0] = start_token
    for t in range(1, ur_maxlen):
        preds = model.predict([enc_input_seq, decoded], verbose=0)
        next_token = np.argmax(preds[:, t-1, :], axis=-1)
        decoded[:, t] = next_token
    return decoded

# Generic wrapper: choose greedy decode based on model type
def greedy_decode(model, enc_input_seq, start_token, end_token, ur_maxlen, model_type="rnn"):
    if model_type in ("transformer",):
        return greedy_decode_transformer(model, enc_input_seq, start_token, end_token, ur_maxlen)
    else:
        return greedy_decode_rnn(model, enc_input_seq, start_token, end_token, ur_maxlen)
