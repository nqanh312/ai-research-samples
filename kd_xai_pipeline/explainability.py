import numpy as np
import torch
from lime.lime_text import LimeTextExplainer


def lime_predict(model, tokenizer, texts, device, max_length=512):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=-1).cpu().numpy()


def collect_unmatched_samples(texts, labels, teacher_model, student_model, tokenizer, device, threshold=0.1):
    explainer = LimeTextExplainer(class_names=[str(i) for i in sorted(set(labels))])
    unmatched = []
    for idx, text in enumerate(texts):
        teacher_pred = np.argmax(lime_predict(teacher_model, tokenizer, [text], device))
        student_pred = np.argmax(lime_predict(student_model, tokenizer, [text], device))
        exp_teacher = explainer.explain_instance(text, lambda x: lime_predict(teacher_model, tokenizer, x, device), num_features=5)
        exp_student = explainer.explain_instance(text, lambda x: lime_predict(student_model, tokenizer, x, device), num_features=5)
        w_teacher, w_student = dict(exp_teacher.as_list()), dict(exp_student.as_list())
        common = set(w_teacher).intersection(w_student)
        diff = np.mean([abs(w_teacher[w] - w_student[w]) for w in common]) if common else 1.0
        if diff > threshold or teacher_pred != student_pred:
            unmatched.append({"index": idx, "label": int(labels[idx]), "teacher_pred": int(teacher_pred), "student_pred": int(student_pred), "lime_diff": float(diff)})
    return unmatched

