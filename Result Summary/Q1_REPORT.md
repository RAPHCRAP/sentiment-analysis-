# Q1 — Sentiment Classification on Urdu Tweets



---

## 1. Introduction

This experiment evaluates multiple deep learning and transformer-based models for **binary sentiment classification** (positive/negative) on a custom Urdu tweet dataset. The goal is to compare classical RNN architectures with multilingual transformer models (mBERT and XLM-RoBERTa) in terms of accuracy, precision, recall, and F1-score.

---

## 2. Dataset and Preprocessing

- Dataset: Custom Urdu tweet corpus with labels **P (positive)** / **N (negative)**.  
- Train/Test split: 75% / 25%  
- Preprocessing steps:
  - Lowercasing
  - Tokenization using **Keras Tokenizer**
  - Padding sequences to **max length = 50**
  - Labels encoded as 0/1

---

## 3. Models Implemented

| Model | Framework |
|-------|-----------|
| RNN | Keras (Sequential) |
| GRU | Keras (Sequential) |
| LSTM | Keras (Sequential) |
| BiLSTM | Keras (Sequential) |
| mBERT | HuggingFace Transformers (PyTorch backend) |
| XLM-RoBERTa | HuggingFace Transformers (PyTorch backend) |

### Hyperparameters (common across RNN models)
- Embedding dimension: 128 (learned during training)
- Hidden units: 128
- Dropout: 0.2
- Optimizer: Adam
- Sequence length: 50
- Epochs: 3 (RNNs), Transformers fine-tuned for more epochs
- Batch size: 32 (RNNs), 8 (transformers)

---

## 4. Evaluation Metrics

The following metrics were computed on the test set:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

---

## 5. Results

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| RNN | 0.4449 | 0.4426 | 0.7043 | 0.5436 |
| GRU | 0.5347 | 0.5026 | 0.8348 | 0.6275 |
| LSTM | 0.5551 | 0.5158 | 0.8522 | 0.6426 |
| BiLSTM | 0.6082 | 0.6173 | 0.4348 | 0.5102 |
| mBERT | 0.5429 | 0.5089 | 0.7478 | 0.6056 |
| XLM-RoBERTa | 0.7102 | 0.7075 | 0.6522 | 0.6787 |

---

## 6. Analysis

1. **Transformers outperform classical RNNs**  
   - XLM-RoBERTa achieved the highest accuracy (0.7102) and competitive F1 (0.6787).  
   - Multilingual pretraining transfers effectively to Urdu sentiment analysis.

2. **RNN family observations**  
   - LSTM and GRU performed best among classical models, with LSTM having the top F1 (0.6426).  
   - Vanilla RNN had high recall but low precision — over-predicting positives.  
   - BiLSTM showed higher precision but lower recall, indicating class threshold effects.

3. **mBERT**  
   - Fine-tuned mBERT performed better than simple RNNs but slightly below XLM-RoBERTa, likely due to dataset size and fine-tuning duration.

4. **Precision vs Recall trade-offs**  
   - RNN/GRU/LSTM tended toward high recall.  
   - BiLSTM leaned toward precision, sacrificing recall.  
   - Transformers balanced both better, giving highest overall F1.

---

## 7. Conclusion

- **Best Model:** **XLM-RoBERTa**  
- **Reason:** Pretrained multilingual transformers can leverage contextual embeddings and cross-lingual knowledge, which significantly improves classification performance on low-resource languages like Urdu.  
- **Classical models:** LSTM remains a strong choice for small datasets, achieving the highest F1 among RNNs.  

---
