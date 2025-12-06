# models/xlmroberta.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_xlm_roberta():
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.eval()
    return tokenizer, model

def encode_transformer(tokenizer, texts, max_len=50):
    encodings = tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else texts,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return encodings

def predict_xlm(tokenizer, model, texts):
    encodings = encode_transformer(tokenizer, texts)
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
    return preds
