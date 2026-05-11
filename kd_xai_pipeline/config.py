from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    raw_excel_path: str = "argument_data_v2.xlsx"
    cleaned_excel_path: str = "cleaned_argumented_data.xlsx"
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    test_csv: str = "test.csv"

    teacher_model_name: str = "vinai/phobert-base"
    teacher_model_dir: str = "./models/PhoBERT"
    best_teacher_dir: str = "./best_model"
    student_models_dir: str = "./student_models"

    batch_size: int = 16
    teacher_batch_size: int = 32
    teacher_epochs: int = 5
    student_epochs: int = 5
    max_length: int = 512
    num_labels: int = 10
    learning_rate: float = 2e-5

    distill_losses: list[str] = field(
        default_factory=lambda: ["kl", "mse", "cosine", "jsd", "supcon", "center", "triplet"]
    )

