import pandas as pd
from ruamel.yaml import YAML

yaml = YAML()


def get_corpus_task(documents_fp):
    df = pd.read_csv(documents_fp, sep="\t")
    return df["text"].to_list()


def get_labels_task(documents_fp):
    df = pd.read_csv(documents_fp, sep="\t")
    return df["sentiment"].to_list()


def get_config_task(config_fp):
    config = None
    with open(config_fp) as f:
        config = yaml.load(f)
    return config


def get_model_hyperparameters_task(config_common, config_expt):
    return {
        "batch_size": config_expt["batch_size"],
        "embed_dim": config_common["embed_dim"],
        "hidden_dim": config_common["hidden_dim"],
        "lstm_n_layers": config_common["lstm_n_layers"],
        "lstm_bidirectional": config_common["lstm_bidirectional"],
        "lstm_dropout_rate": config_common["lstm_dropout_rate"],
    }
