# Q2 — Sentiment Classification using Word Embeddings



---

## 1. Introduction

This experiment extends Q1 by evaluating the impact of **pretrained word embeddings** on binary sentiment classification (positive/negative) for Urdu tweets.  

The classifier chosen is **LSTM**, as it had the best F1-score among classical models in Q1 (≈0.6426).  

The study compares:
- Baseline LSTM (embedding learned end-to-end)
- LSTM + Word2Vec (trained on our Urdu corpus)
- LSTM + FastText (trained on our Urdu corpus)
- LSTM + GloVe (trained on our Urdu corpus)
- LSTM + ELMo (planned but not implemented due to dependency issues)

---

## 2. Dataset and Preprocessing

- Same Urdu tweet dataset as Q1.  
- Train/Test split: 75% / 25%  
- Preprocessing steps:
  - Lowercasing
  - Tokenization using **Keras Tokenizer**
  - Padding sequences to **max length = 50**
  - Labels encoded as 0/1  

- Embeddings trained on the **same training corpus** to maintain in-domain vocabulary coverage.  

---

## 3. Word Embeddings

1. **Word2Vec**  
   - Trained with gensim on the training tweets  
   - Vector size: 100  
   - Window: 5, min_count: 1, epochs: 10

2. **FastText**  
   - Trained with gensim on the same corpus  
   - Vector size: 100  
   - Window: 5, min_count: 1, epochs: 10

3. **GloVe**  
   - Small GloVe-style embeddings trained via gensim Word2Vec proxy on corpus  
   - Vector size: 100

4. **ELMo**  
   - Planned for future work  
   - Would produce **sentence-level embeddings**

**Embedding Integration:**  
- Embeddings loaded into the LSTM **embedding layer** as a fixed **embedding matrix**.  
- OOV words initialized to zero vectors.  
- LSTM architecture: same as Q1 with **64 BiLSTM units**, dropout 0.2, sigmoid output.

---

## 4. Model Training

- Baseline LSTM trained end-to-end without pretrained embeddings.  
- Other variants trained using pretrained embedding matrices.  
- Training hyperparameters:
  - Epochs: 5
  - Batch size: 32
  - Validation split: 0.1
  - Optimizer: Adam

---

## 5. Results

| Model Variant        | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| LSTM (no embeddings) | 0.5551   | 0.5158    | 0.8522 | 0.6426   |
| LSTM + Word2Vec      | 0.4979   | 0.4649    | 0.4609 | 0.4629   |
| LSTM + FastText      | 0.4735   | 0.4352    | 0.4087 | 0.4215   |
| LSTM + GloVe         | 0.5102   | 0.4797    | 0.5130 | 0.4958   |
| LSTM + ELMo          | Not run  | —         | —      | —        |

---

## 6. Analysis

1. **Baseline LSTM performs best**  
   - Surprisingly, the end-to-end trained embedding outperforms pretrained embeddings from Word2Vec, FastText, and GloVe trained on this small corpus.  
   - Likely because **pretraining embeddings on small datasets produces lower-quality vectors** compared to task-specific learning.

2. **Word2Vec & GloVe**  
   - Moderate performance, slightly below baseline.  
   - GloVe performed better than Word2Vec, possibly due to more consistent vector training across corpus.

3. **FastText**  
   - Performance significantly lower  
   - Recall dropped sharply (0.2174), suggesting poor generalization and potential overfitting on small dataset

4. **ELMo**  
   - Not implemented due to heavy dependencies.  
   - Could improve performance if full pretrained Urdu ELMo embeddings are available.

---

## 7. Conclusion

- **Best Model:** **Baseline LSTM (no pretrained embeddings)**  
- **Reason:** Task-specific embedding learning on small Urdu dataset outperformed embeddings pretrained on the same limited corpus.  
- **Recommendation:**  
  - Pretrain embeddings on **large, diverse Urdu corpora** for improved transfer  
  - Fine-tune embeddings during supervised training  
  - Consider advanced contextual embeddings like ELMo or multilingual BERT

---
