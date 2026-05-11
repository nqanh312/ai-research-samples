import os
import time

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from .config import PipelineConfig


def _encode(df: pd.DataFrame, tokenizer, max_length: int):
    return tokenizer(list(df["text"]), truncation=True, padding=True, max_length=max_length, return_tensors="pt")


def _to_dataset(encodings, labels):
    return Dataset.from_dict(
        {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"], "labels": labels.tolist()}
    )


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def train_teacher(config: PipelineConfig) -> None:
    train_df = pd.read_csv(config.train_csv)
    val_df = pd.read_csv(config.val_csv)
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    train_dataset = _to_dataset(_encode(train_df, tokenizer, config.max_length), train_df["label"])
    val_dataset = _to_dataset(_encode(val_df, tokenizer, config.max_length), val_df["label"])

    model = AutoModelForSequenceClassification.from_pretrained(config.teacher_model_name, num_labels=config.num_labels)
    model.resize_token_embeddings(len(tokenizer))

    args = TrainingArguments(
        output_dir="./results/PhoBERT",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=config.teacher_batch_size,
        per_device_eval_batch_size=config.teacher_batch_size,
        num_train_epochs=config.teacher_epochs,
        fp16=True,
        report_to="none",
        logging_steps=100,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=val_dataset, tokenizer=tokenizer, compute_metrics=_compute_metrics)
    start = time.time()
    trainer.train()
    print(f"Teacher train time: {time.time() - start:.2f}s")

    os.makedirs(config.teacher_model_dir, exist_ok=True)
    model.save_pretrained(config.teacher_model_dir)
    tokenizer.save_pretrained(config.teacher_model_dir)


def evaluate_teacher(config: PipelineConfig) -> dict:
    model_path = config.best_teacher_dir if os.path.exists(config.best_teacher_dir) else config.teacher_model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=config.num_labels).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_df = pd.read_csv(config.test_csv)
    enc = _encode(test_df, tokenizer, config.max_length)
    test_ds = Dataset.from_dict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": torch.tensor(test_df["label"])})
    loader = DataLoader(test_ds.with_format("torch"), batch_size=config.batch_size)

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
            y_pred.extend(logits.argmax(dim=-1).cpu().numpy())
            y_true.extend(batch["labels"].cpu().numpy())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }

