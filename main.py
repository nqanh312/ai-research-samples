import argparse

from kd_xai_pipeline.config import PipelineConfig
from kd_xai_pipeline.preprocessing import run_preprocessing
from kd_xai_pipeline.student import evaluate_students, train_distilled_students
from kd_xai_pipeline.teacher import evaluate_teacher, train_teacher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KD-XAI modular pipeline")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["preprocess", "train_teacher", "eval_teacher", "distill_student", "eval_student", "all"],
        help="Pipeline stage to run",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig()

    if args.stage in {"preprocess", "all"}:
        run_preprocessing(config)

    if args.stage in {"train_teacher", "all"}:
        train_teacher(config)

    if args.stage in {"eval_teacher", "all"}:
        print("Teacher metrics:", evaluate_teacher(config))

    if args.stage in {"distill_student", "all"}:
        train_distilled_students(config)

    if args.stage in {"eval_student", "all"}:
        results = evaluate_students(config)
        if results.empty:
            print("No student model found in ./student_models.")
        else:
            print(results.to_string(index=False))


if __name__ == "__main__":
    main()

