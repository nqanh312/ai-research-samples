import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .config import PipelineConfig


class SmartStudentPhoBERT(nn.Module):
    def __init__(self, pretrained_model: str = "vinai/phobert-base", num_labels: int = 10, num_student_layers: int = 4):
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_model, output_hidden_states=True)
        self.config = config
        backbone = AutoModel.from_pretrained(pretrained_model, config=config)
        backbone.encoder.layer = nn.ModuleList(backbone.encoder.layer[:num_student_layers])
        self.backbone = backbone

        hidden_size = config.hidden_size
        self.attn_pool = nn.Sequential(nn.Linear(hidden_size, 1), nn.Softmax(dim=1))
        self.gate = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())
        self.projection = nn.Sequential(nn.Linear(hidden_size, 512), nn.ReLU(), nn.Dropout(0.1))
        self.classifier = nn.Linear(512, num_labels)

    def forward(self, input_ids, attention_mask, output_attentions=False, output_hidden_states=False):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = outputs.last_hidden_state
        cls_token = hidden_states[:, 0, :]
        attn_scores = self.attn_pool(hidden_states)
        pooled_token = torch.sum(attn_scores * hidden_states, dim=1)
        fused = self.gate(torch.cat([cls_token, pooled_token], dim=1)) * cls_token + (1 - self.gate(torch.cat([cls_token, pooled_token], dim=1))) * pooled_token
        logits = self.classifier(self.projection(fused))
        return {"logits": logits, "cls_rep": fused}


class GeneralDistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=2.0, loss_type="kl"):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.loss_type = loss_type
        self.hard_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        t = self.temperature
        soft_student = F.log_softmax(student_logits / t, dim=-1)
        soft_teacher = F.softmax(teacher_logits / t, dim=-1)
        if self.loss_type == "kl":
            soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * t**2
        elif self.loss_type == "mse":
            soft_loss = F.mse_loss(student_logits, teacher_logits)
        elif self.loss_type == "cosine":
            soft_loss = 1 - F.cosine_similarity(student_logits, teacher_logits, dim=-1).mean()
        else:
            m = 0.5 * (soft_teacher + torch.exp(soft_student))
            soft_loss = 0.5 * (F.kl_div(soft_student, m, reduction="batchmean") + F.kl_div(torch.log(m), soft_teacher, reduction="batchmean"))
        return self.alpha * soft_loss + (1 - self.alpha) * self.hard_loss(student_logits, labels)


def _encode_data(df: pd.DataFrame, tokenizer, max_length: int):
    return tokenizer(list(df["text"]), truncation=True, padding=True, max_length=max_length, return_tensors="pt")


def _build_train_loader(config: PipelineConfig):
    train_df = pd.read_csv(config.train_csv)
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
    encodings = _encode_data(train_df, tokenizer, config.max_length)
    labels = torch.tensor(train_df["label"].values)
    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True), tokenizer


def train_distilled_students(config: PipelineConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.student_models_dir, exist_ok=True)
    train_loader, tokenizer = _build_train_loader(config)

    teacher_path = config.best_teacher_dir if os.path.exists(config.best_teacher_dir) else config.teacher_model_dir
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path, num_labels=config.num_labels).to(device).eval()

    for loss_name in config.distill_losses:
        save_dir = os.path.join(config.student_models_dir, f"student_{loss_name}")
        os.makedirs(save_dir, exist_ok=True)

        student = SmartStudentPhoBERT(num_labels=config.num_labels).to(device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)
        criterion = GeneralDistillationLoss(alpha=0.7, temperature=2.0, loss_type=loss_name if loss_name in {"kl", "mse", "cosine"} else "jsd")

        for epoch in range(config.student_epochs):
            student.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for input_ids, attention_mask, labels in train_loader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                with torch.no_grad():
                    teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
                outputs = student(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs["logits"], teacher_logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_correct += (outputs["logits"].argmax(dim=-1) == labels).sum().item()
                total_samples += labels.size(0)
            print(f"[{loss_name}] Epoch {epoch+1}/{config.student_epochs} - Loss: {total_loss/len(train_loader):.4f} - Acc: {total_correct/max(total_samples,1):.2%}")

        torch.save(student.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as fh:
            fh.write(student.config.to_json_string())


def evaluate_students(config: PipelineConfig) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(config.test_csv)
    tokenizer = AutoTokenizer.from_pretrained(config.best_teacher_dir if os.path.exists(config.best_teacher_dir) else config.teacher_model_name)
    encodings = _encode_data(test_df, tokenizer, config.max_length)
    labels = torch.tensor(test_df["label"].values)
    test_loader = DataLoader(TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels), batch_size=config.batch_size)

    rows = []
    for loss_name in config.distill_losses:
        model_path = os.path.join(config.student_models_dir, f"student_{loss_name}", "pytorch_model.bin")
        if not os.path.exists(model_path):
            continue
        model = SmartStudentPhoBERT(num_labels=config.num_labels).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        y_true, y_pred, infer_times = [], [], []
        with torch.no_grad():
            for input_ids, attention_mask, batch_labels in test_loader:
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                start = time.time()
                logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
                infer_times.append(time.time() - start)
                preds = logits.argmax(dim=-1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(batch_labels.numpy())

        rows.append(
            {
                "loss": loss_name,
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "avg_infer_time": float(np.mean(infer_times)) if infer_times else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(by="f1", ascending=False) if rows else pd.DataFrame()

