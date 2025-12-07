# models/xlmroberta.py
import torch
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

def load_xlm_roberta(X_train=None, y_train=None, X_val=None, y_val=None, epochs=3):
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)

    if X_train is not None and y_train is not None:
        train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
        val_encodings   = tokenizer(list(X_val), truncation=True, padding=True)

        train_dataset = SentimentDataset(train_encodings, list(y_train))
        val_dataset   = SentimentDataset(val_encodings, list(y_val))

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./bert_results",
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy="epoch",          # ← updated argument name
            save_strategy="epoch",
            logging_steps=10,
            learning_rate=2e-5,
            disable_tqdm=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()

    return tokenizer, model

def predict_xlm(tokenizer, model, texts):
    model.eval()
    inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).numpy()
    return preds
