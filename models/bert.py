# models/bert.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_mbert():
    """
    Load multilingual BERT model and tokenizer for sequence classification.
    Uses PyTorch backend to avoid TF/Keras issues.
    """
    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.eval()  # set to eval mode by default
    return tokenizer, model

def encode_transformer(tokenizer, texts, max_len=50):
    """
    Tokenize texts and return PyTorch tensors
    """
    encodings = tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else texts,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return encodings

def predict_mbert(tokenizer, model, texts):
    """
    Predict sentiment using mBERT model
    """
    encodings = encode_transformer(tokenizer, texts)
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
    return preds
