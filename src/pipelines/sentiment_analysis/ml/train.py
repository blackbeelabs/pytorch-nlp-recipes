from functools import partial
import json
from os import path

import torch
from torch.utils.data import DataLoader
import pendulum

from utils.io_utils import *
from utils.text_utils import *
from model.model import *
from tasks.tasks import *


def _get_project_dir_folder():
    return path.dirname(
        path.dirname(path.dirname(path.dirname(path.dirname(path.dirname(__file__)))))
    )


def _construct_report(config, losses, now, later, vocab_size, model):
    report = {}
    report["name"] = config["name"]
    report["expriment"] = config["experiment"]
    report = report | config["params"]["common"]
    report = report | config["params"]["train"]

    report["pipeline_time_start"] = now.to_datetime_string()
    report["pipeline_time_end"] = later.to_datetime_string()
    report["training_time_taken_minutes"] = (later - now).in_minutes()
    report["training_loss_curve"] = str(losses)
    report["model_architecture"] = str(model.eval())
    report["model_count_parameters"] = count_parameters(model)
    report["model_vocab_size"] = vocab_size
    return report


def main():

    now = pendulum.now()
    model_timestamp = now.format("YYYYMMDDTHHmmss")
    torch.manual_seed(42)
    device = get_device()

    workflow = "model"
    experiment = "baseline"
    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")
    # Input
    documents_fp = path.join(
        ASSETS_FP, "datasets", "sentiment_analysis", "datamart", "train.csv"
    )
    config_fp = path.join(
        ASSETS_FP, "config", "sentiment_analysis", f"config-{experiment}.yaml"
    )
    # Output
    model_vocabulary_fp = path.join(
        ASSETS_FP,
        "datasets",
        "sentiment_analysis",
        "model",
        f"vocabulary-{workflow}-{experiment}-{model_timestamp}.csv",
    )
    model_fp = path.join(
        ASSETS_FP,
        "models",
        "sentiment_analysis",
        f"{workflow}-{experiment}-{model_timestamp}.pkl",
    )
    model_json_fp = path.join(
        ASSETS_FP,
        "models",
        "sentiment_analysis",
        f"{workflow}profile-{experiment}-{model_timestamp}.json",
    )

    # Corpus
    corpus = get_corpus_task(documents_fp)
    # Labels
    labels = get_labels_task(documents_fp)
    # Config
    config = get_config_task(config_fp)
    config_common = config.get("params").get("common")
    config_train = config.get("params").get("train")
    # Vocabulary
    word_to_idx_dict, vocab_size = construct_word_to_idx(corpus)
    save_vocabulary_to_idx(word_to_idx_dict, model_vocabulary_fp)
    # Corpus to index sequence
    word_idx_sequences = transform_corpus_to_idx_sequences(
        corpus,
        word_to_idx_dict,
    )
    # Data loader
    data = list(zip(word_idx_sequences, labels))
    collate_fn = partial(
        custom_collate_fn_for_variable_seq_length,
        word_to_idx=word_to_idx_dict,
        device=device,
    )
    loader = DataLoader(
        data,
        batch_size=config_train["batch_size"],
        shuffle=config_train["shuffle"],
        collate_fn=collate_fn,
    )
    # Model
    model_hyperparameters = get_model_hyperparameters_task(config_common, config_train)
    model = SentimentClassifierBaseline(
        model_hyperparameters,
        vocab_size,
    )
    # Train model
    optimizer = optimizer_obj(
        model, config_train["optimizer"], config_train["learning_rate"]
    )
    # Train
    losses = train_sentiment_analysis(
        bce_loss_function,
        optimizer,
        model,
        loader,
        num_epochs=config_train["epochs"],
    )
    later = pendulum.now()
    # Save trained model & environment
    torch.save(model.state_dict(), model_fp)
    j = _construct_report(config, losses, now, later, vocab_size, model)
    with open(model_json_fp, "w") as f:
        json.dump(j, f, ensure_ascii=False, indent=4)
    print(f"Done. Timestamp = {model_timestamp}")


if __name__ == "__main__":
    main()
