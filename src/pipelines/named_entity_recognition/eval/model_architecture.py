import json
from os import path
from functools import partial

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

    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")

    experiment_mode = "model"
    experiment = "baseline"
    experiment_timestamp = "20240228T151708"

    config_baseline_fp = path.join(ASSETS_FP, "config", f"config-{experiment}.yaml")
    model_fp = path.join(
        ASSETS_FP,
        "models",
        f"{experiment_mode}-{experiment}-{experiment_timestamp}.pkl",
    )
    vocabulary_fp = path.join(
        ASSETS_FP, "datasets", "model", f"vocabulary-{experiment}.csv"
    )
    labels_fp = path.join(ASSETS_FP, "datasets", "datamart", "labels.csv")
    # Labels
    labels_df = pd.read_csv(labels_fp, sep="\t")
    count_unique_labels = labels_df.shape[0]
    # Config
    config = None
    with open(config_baseline_fp) as f:
        config = yaml.load(f)
    config_common = config.get("params").get("common")
    config_validate = config.get("params").get("validate")
    # Vocabulary
    word_to_idx_dict = load_vocabulary_to_idx(vocabulary_fp)
    vocab_size = len(word_to_idx_dict)

    # Load model
    model_hyperparameters = {
        "batch_size": config_validate["batch_size"],
        "window_size": config_common["window_size"],
        "embed_dim": config_common["embed_dim"],
        "hidden_dim": config_common["embed_dim"],
        "freeze_embeddings": config_common["freeze_embeddings"],
    }
    model = WordWindowMulticlassClassifierBaseline(
        model_hyperparameters, vocab_size, num_classes=count_unique_labels
    )
    model.load_state_dict(torch.load(model_fp))

    i, j = count_parameters(model)
    print(i)
    print(j)


if __name__ == "__main__":
    main()
