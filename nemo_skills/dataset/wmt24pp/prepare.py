import argparse
import importlib.util
import json
import tempfile
import urllib.request
from pathlib import Path

from langcodes import Language

from datasets import load_dataset
from tqdm import tqdm

LANG2CODE = {"de": "de_DE", "es": "es_MX", "fr": "fr_FR", "it": "it_IT", "ja": "ja_JP"}


def make_prompt(source_text, target_lang):
    lang_name = Language(target_lang).display_name()
    prompt = f"Translate the following segment into {lang_name}, without additional explanation.\n\n{source_text}"
    return prompt


def write_data_to_file(output_file, datasets, tgt_languages):
    with open(output_file, "wt", encoding="utf-8") as fout:     
        for tgt_lang in tgt_languages:
                for src, tgt in zip(datasets[tgt_lang]["source"], datasets[tgt_lang]["target"]):
                    json_dict = {
                        "question": make_prompt(src, tgt_lang),
                        "text": src,
                        "translation": tgt,
                        "source_language": "en",
                        "target_language": tgt_lang
                    }
                    json.dump(json_dict, fout)
                    fout.write("\n")


def main(args):

    datasets = {}
    for lang in args.target_languages:
        lang_code = f"en-{LANG2CODE[lang]}"
        datasets[lang] = load_dataset("google/wmt24pp", lang_code)["train"]

    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{args.split}.jsonl"
    write_data_to_file(
        output_file,
        datasets,
        tgt_languages=args.target_languages
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("test"), help="Dataset split to process.")
    parser.add_argument(
        "--target_languages", default=["de", "es", "fr", "it", "ja"],
        nargs="+", help="Languages to translate to."
    )
    args = parser.parse_args()
    main(args)
