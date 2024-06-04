from functools import partial
import json
from os import path

import torch
from torch.utils.data import DataLoader
from ruamel.yaml import YAML
import pendulum

from utils.io_utils import *
from utils.text_utils import *
from model.model import *

yaml = YAML()


def _get_project_dir_folder():
    return path.dirname(path.dirname(path.dirname(path.dirname(__file__))))


def main():

    experiment = "baseline"
    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")
    device = get_device()

    documents_fp = path.join(ASSETS_FP, "datasets", "model", f"train-{experiment}.csv")
    labels_fp = path.join(ASSETS_FP, "datasets", "datamart", "labels.csv")
    config_fp = path.join(ASSETS_FP, "config", f"config-{experiment}.yaml")

    model_vocabulary_fp = path.join(
        ASSETS_FP, "datasets", "model", f"vocabulary-{experiment}.csv"
    )

    # Corpus
    df = pd.read_csv(documents_fp, sep="\t")
    print(f"df.shape={df.shape}")
    corpus = df["tokens"]
    # Labels
    labels_df = pd.read_csv(labels_fp, sep="\t")
    labels = df["onehot_labels"].to_list()
    count_unique_labels = labels_df.shape[0]
    # Config
    config = None
    with open(config_fp) as f:
        config = yaml.load(f)
    config_common = config.get("params").get("common")
    config_train = config.get("params").get("train")

    # Vocabulary
    word_to_idx_dict = construct_word_to_idx(corpus)
    vocab_size = len(word_to_idx_dict)
    print(f"vocab_size={vocab_size}")

    # Data Loader
    word_idx_sequences, onehot_labels = transform_corpus_to_idx_word_windows(
        corpus, labels, word_to_idx_dict, pad_window_size=config_common["window_size"]
    )
    data = list(zip(word_idx_sequences, onehot_labels))
    collate_fn = partial(custom_collate_fn, device=device)
    loader = DataLoader(
        data,
        batch_size=config_train["batch_size"],
        shuffle=config_train["shuffle"],
        collate_fn=collate_fn,
    )

    for xB, yB, lB in loader:
        print(xB.size())
        print(yB.size())
        print(lB.size())
        print()


if __name__ == "__main__":
    main()
