import os
import regex as re

import pandas as pd
from pyvi import ViTokenizer
from sklearn.model_selection import train_test_split

from .config import PipelineConfig


def remove_html(txt: str) -> str:
    return re.sub(r"<[^>]*>", "", txt)


def _load_diacritic_map() -> dict[str, str]:
    char1252 = (
        "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|"
        "ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|"
        "À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|"
        "Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ"
    ).split("|")
    charutf8 = (
        "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|"
        "ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|"
        "À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|"
        "Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ"
    ).split("|")
    return {char1252[i]: charutf8[i] for i in range(len(char1252))}


DIACRITIC_MAP = _load_diacritic_map()


def convert_unicode(txt: str) -> str:
    pattern = (
        r"à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|"
        r"ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|"
        r"À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|"
        r"Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ"
    )
    return re.sub(pattern, lambda x: DIACRITIC_MAP[x.group()], txt)


def text_preprocess(text: str) -> str:
    text = remove_html(text)
    text = convert_unicode(text).lower()
    text = re.sub(
        r"[^a-zA-Z0-9àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ\s]",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return ViTokenizer.tokenize(text)


def run_preprocessing(config: PipelineConfig) -> None:
    df = pd.read_excel(config.raw_excel_path, sheet_name=0, header=None)
    df.columns = ["Description", "Label"] + [f"Col_{i}" for i in range(3, df.shape[1] + 1)]
    df.drop_duplicates(inplace=True)
    df["Description"] = df["Description"].fillna("")
    df["Cleaned_Description"] = df["Description"].apply(text_preprocess)
    df.to_excel(config.cleaned_excel_path, index=False)

    cleaned = pd.read_excel(config.cleaned_excel_path)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        cleaned["Cleaned_Description"],
        cleaned["Label"],
        test_size=0.4,
        stratify=cleaned["Label"],
        random_state=42,
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    pd.DataFrame({"text": train_texts, "label": train_labels}).reset_index(drop=True).to_csv(
        config.train_csv, index=False, encoding="utf-8"
    )
    pd.DataFrame({"text": val_texts, "label": val_labels}).reset_index(drop=True).to_csv(
        config.val_csv, index=False, encoding="utf-8"
    )
    pd.DataFrame({"text": test_texts, "label": test_labels}).reset_index(drop=True).to_csv(
        config.test_csv, index=False, encoding="utf-8"
    )

    print("Preprocessing done:")
    print(f"- cleaned excel: {os.path.abspath(config.cleaned_excel_path)}")
    print(f"- train/val/test csv: {config.train_csv}, {config.val_csv}, {config.test_csv}")

