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
    return path.dirname(
        path.dirname(path.dirname(path.dirname(path.dirname(path.dirname(__file__)))))
    )


def _construct_report(config, now, later, model):
    report = {}
    report["name"] = config["name"]
    expt = config["experiment"]
    report["expriment"] = f"build-{expt}"
    report = report | config["params"]["common"]
    report = report | config["params"]["train"]

    report["pipeline_time_start"] = now.to_datetime_string()
    report["model_architecture"] = str(model.eval())
    report["model_count_parameters"] = count_parameters(model)
    return report


def main():

    now = pendulum.now()
    model_timestamp = now.format("YYYYMMDDTHHmmss")
    torch.manual_seed(42)
    device = get_device()

    experiment = "baseline"
    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")
    # Input
    documents_fp = path.join(
        ASSETS_FP,
        "datasets",
        "named_entity_recognition",
        "model",
        f"train-{experiment}.csv",
    )
    labels_fp = path.join(
        ASSETS_FP, "datasets", "named_entity_recognition", "datamart", "labels.csv"
    )
    config_fp = path.join(
        ASSETS_FP, "config", "named_entity_recognition", f"config-{experiment}.yaml"
    )
    model_vocabulary_fp = path.join(
        ASSETS_FP,
        "datasets",
        "model",
        "named_entity_recognition",
        f"vocabulary-{experiment}.csv",
    )
    # Output
    model_fp = path.join(
        ASSETS_FP,
        "models",
        "named_entity_recognition",
        f"build-{experiment}-{model_timestamp}.pkl",
    )
    model_json_fp = path.join(
        ASSETS_FP,
        "models",
        "named_entity_recognition",
        f"buildprofile-{experiment}-{model_timestamp}.json",
    )

    # Corpus
    df = pd.read_csv(documents_fp, sep="\t")
    corpus = df["tokens"]
    corpus = corpus[:10]
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
    word_to_idx_dict, vocab_size = construct_word_to_idx(corpus)

    # Model
    model_hyperparameters = {
        "batch_size": config_train["batch_size"],
        "window_size": config_common["window_size"],
        "embed_dim": config_common["embed_dim"],
        "hidden_dim": config_common["embed_dim"],
        "freeze_embeddings": config_common["freeze_embeddings"],
    }
    model = WordWindowMulticlassClassifierBaseline(
        model_hyperparameters, vocab_size, num_classes=count_unique_labels
    )
    later = pendulum.now()

    # Save trained model & environment
    torch.save(model.state_dict(), model_fp)
    j = _construct_report(config, now, later, model)
    with open(model_json_fp, "w") as f:
        json.dump(j, f, ensure_ascii=False, indent=4)
    print(f"Done. Timestamp = {model_timestamp}")


if __name__ == "__main__":
    main()
