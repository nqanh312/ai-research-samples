# KD-XAI Pipeline (Modular)

This project is split into small modules so readers can follow the full flow quickly.

## Structure

- `main.py`: single entry point to run each stage.
- `kd_xai_pipeline/config.py`: shared configuration.
- `kd_xai_pipeline/preprocessing.py`: clean text and generate `train/val/test`.
- `kd_xai_pipeline/teacher.py`: train and evaluate teacher model.
- `kd_xai_pipeline/student.py`: distill student models and evaluate them.
- `kd_xai_pipeline/explainability.py`: LIME helper utilities.
- `knowledge_distillation_xai.py`: legacy monolithic script (kept for reference).

## End-to-end flow

1. Preprocess source Excel into cleaned text and split CSV files.
2. Train teacher model.
3. Distill students from teacher.
4. Evaluate teacher and students.
5. (Optional) Run explainability utilities.

## Run

From `ai-research-samples`:

```bash
python main.py --stage preprocess
python main.py --stage train_teacher
python main.py --stage eval_teacher
python main.py --stage distill_student
python main.py --stage eval_student
```

Run everything:

```bash
python main.py --stage all
```

## Notes

- Default paths and hyperparameters are in `kd_xai_pipeline/config.py`.
- Teacher path priority in evaluation/distillation: `./best_model`


