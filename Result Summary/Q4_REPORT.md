# Q4 — Advanced Sentiment Classification using Transformer Models



---

## 1. Introduction

This experiment focuses on **binary sentiment classification** using **pretrained multilingual transformer models**, specifically:

- **mBERT (Multilingual BERT)**
- **XLM-RoBERTa**

Goal: Evaluate fine-tuned transformer models on the **Urdu sentiment dataset** and compare their performance against classical deep learning models (RNN, LSTM, BiLSTM).

---

## 2. Dataset and Preprocessing

- Dataset: Urdu Sentiment Corpus (same as Q1/Q2)
- Labels: Positive / Negative
- Train/Test split: 75% / 25%
- Preprocessing:
  - Lowercasing
  - Tokenization handled by HuggingFace tokenizer for each model
- Maximum sequence length: 50 tokens
- Padding/truncation applied to fit model input requirements

---

## 3. Model Fine-Tuning

| Model           | Architecture                  | Batch Size | Learning Rate | Epochs | Notes |
|-----------------|-------------------------------|------------|---------------|--------|-------|
| mBERT           | Transformer (12-layer BERT)   | 8          | 5e-5          | 3      | HuggingFace `BertForSequenceClassification` |
| XLM-RoBERTa     | Transformer (24-layer RoBERTa)| 8          | 5e-5          | 3      | HuggingFace `XLMRobertaForSequenceClassification` |

Training involved:
- AdamW optimizer
- Cross-entropy loss
- Gradient clipping for stability
- Validation split: 10% of training data

---

## 4. Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score

**Evaluation performed on the test set.**

---

## 5. Results

| Model           | Accuracy | Precision | Recall | F1    |
|-----------------|----------|-----------|--------|-------|
| mBERT           | 0.5429   | 0.5089    | 0.7478 | 0.6056 |
| XLM-RoBERTa     | 0.7102   | 0.7075    | 0.6522 | 0.6787 |

**Observation:**  
- XLM-RoBERTa outperformed mBERT in both accuracy and F1-score.
- Both models improved over classical RNN-based models (RNN, LSTM, BiLSTM).

---

## 6. Analysis

1. **mBERT**
   - Pretrained on 104 languages including Urdu
   - Fine-tuning helped transfer multilingual knowledge
   - F1 ≈ 0.61, good recall but lower precision

2. **XLM-RoBERTa**
   - Pretrained on a large multilingual corpus
   - Handles low-resource languages better than mBERT
   - Highest accuracy and F1 among all models
   - More robust on noisy or small datasets

3. **Comparison with Classical Models**
   - Classical RNNs struggled to capture nuanced Urdu sentiment
   - Transformers leverage large-scale pretraining for contextual understanding
   - Recall and precision trade-offs differ due to pretraining and tokenization strategies

---

## 7. Sample Predictions

| Urdu Tweet                     | True Label | mBERT Prediction | XLM-RoBERTa Prediction |
|--------------------------------|------------|-----------------|-----------------------|
| "میں اس فلم سے خوش ہوں۔"       | Positive   | Positive        | Positive              |
| "یہ سروس بہت خراب ہے۔"         | Negative   | Positive        | Negative              |
| "فلم کی کہانی زبردست تھی۔"     | Positive   | Positive        | Positive              |
| "میں دوبارہ یہ جگہ نہیں آؤں گا۔" | Negative   | Negative        | Negative              |

*(Predictions illustrative; exact results depend on training)*

---

## 8. Conclusion

- **Best-performing model:** XLM-RoBERTa
- **Reasons for superior performance:**
  - Larger multilingual pretraining corpus
  - Robust tokenization and subword handling
  - Better contextual embeddings for low-resource languages
- mBERT performed reasonably well but slightly behind XLM-RoBERTa
- Classical RNNs underperformed due to limited dataset size and inability to capture contextual semantics

**Recommendation:**  
For **Urdu sentiment classification**, fine-tuned **XLM-RoBERTa** is recommended. For small datasets, consider **data augmentation** or **multilingual pretraining** to improve classical models.

---
