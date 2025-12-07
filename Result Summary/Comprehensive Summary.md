# NLP Assignment — Comprehensive Report

---

## Q1 — Sentiment Classification (Binary)

### 1.1 Introduction
Binary sentiment classification on Urdu text using classical deep learning models and multilingual transformers.

### 1.2 Models & Hyperparameters
- **RNN, GRU, LSTM, BiLSTM**: Keras; embedding=128, hidden=128, dropout=0.2, maxlen=50  
- **mBERT & XLM-RoBERTa**: HuggingFace Transformers; batch=8, learning rate=5e-5  

### 1.3 Results

| Model           | Accuracy | Precision | Recall | F1    |
|-----------------|----------|-----------|--------|-------|
| RNN             | 0.4449   | 0.4426    | 0.7043 | 0.5436 |
| GRU             | 0.5347   | 0.5026    | 0.8348 | 0.6275 |
| LSTM            | 0.5551   | 0.5158    | 0.8522 | 0.6426 |
| BiLSTM          | 0.6082   | 0.6173    | 0.4348 | 0.5102 |
| mBERT           | 0.5429   | 0.5089    | 0.7478 | 0.6056 |
| XLM-RoBERTa     | 0.7102   | 0.7075    | 0.6522 | 0.6787 |

**Insight:** XLM-RoBERTa outperforms all models due to large-scale multilingual pretraining. Classical RNNs are limited by small dataset size and lack of contextual embeddings.

---

## Q2 — Sentiment Classification using Embeddings

### 2.1 Introduction
We used **LSTM** (best classical F1 from Q1) and tested different **pretrained embeddings** trained on the Urdu corpus.

### 2.2 Results

| Model Variant       | Accuracy | Precision | Recall | F1    |
|--------------------|----------|-----------|--------|-------|
| LSTM (no embeddings)| 0.5551   | 0.5158    | 0.8522 | 0.6426 |
| LSTM + Word2Vec    | 0.4979   | 0.4649    | 0.4609 | 0.4629 |
| LSTM + FastText    | 0.4734   | 0.4352    | 0.4087 | 0.4215 |
| LSTM + GloVe       | 0.5102   | 0.4796    | 0.5130 | 0.4957 |

**Insight:**  
- Task-specific LSTM embeddings trained end-to-end outperform pretrained embeddings on this **small Urdu corpus**.  
- Pretrained embeddings require large corpora; small corpus embeddings can be noisy and underperform.

---

## Q3 — Seq2Seq Models: English→Urdu Translation

### 3.1 Models & Hyperparameters

| Model       | Embedding | Hidden | Epochs | Batch |
|-------------|-----------|--------|--------|-------|
| RNN         | 256       | 512    | 50     | 64    |
| BiRNN       | 256       | 512    | 50     | 64    |
| LSTM        | 256       | 512    | 50     | 64    |
| BiLSTM      | 256       | 512    | 50     | 64    |
| Transformer | 256       | 512    | 50     | 64    |

### 3.2 BLEU Scores

| Model       | BLEU Score |
|-------------|------------|
| RNN         | 0.00       |
| BiRNN       | 18.996     |
| LSTM        | 12.703     |
| BiLSTM      | 17.400     |
| Transformer | 18.996     |

**Insight:**  
- Vanilla RNNs fail due to inability to capture long-term dependencies.  
- BiRNN and Transformer models achieve the highest BLEU (~19).  
- BiLSTM captures context better than LSTM and approaches BiRNN performance.  
- Small dataset limits Transformer potential.

---
## Q4 — RNN Seq2Seq Translation: Random vs GloVe Embeddings

### 4.1 Introduction
Compare **randomly initialized embeddings** vs **pretrained GloVe embeddings** in an RNN Seq2Seq model for English → Urdu translation. Evaluate impact on translation quality and convergence.

### 4.2 Models & Hyperparameters
- **RNN Seq2Seq (Random embeddings)**: Vanilla RNN, embedding=128, hidden=256, epochs=3, batch=32  
- **RNN Seq2Seq (GloVe embeddings)**: Vanilla RNN, embedding=100, hidden=256, epochs=3, batch=32, embeddings frozen  

### 4.3 Results

| Model               | Accuracy | Loss    |
|---------------------|---------|--------|
| Random Embeddings    | 0.9524  | 0.3617 |
| GloVe Embeddings     | 0.9799  | 0.1493 |

**Insight:** Pretrained GloVe embeddings slightly improve accuracy and significantly reduce loss, indicating faster convergence and better semantic understanding. Random embeddings still perform well on small datasets, but pretrained embeddings are advantageous for larger corpora or more complex tasks.
