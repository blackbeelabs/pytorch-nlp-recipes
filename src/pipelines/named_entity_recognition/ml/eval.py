import argparse
import json
from os import path
from functools import partial

import torch
from torch.utils.data import DataLoader
from ruamel.yaml import YAML
import pendulum

from src.utils.io_utils import *
from src.utils.text_utils import *
from src.model.model import *

yaml = YAML()


def _get_project_dir_folder():
    project_root = path.dirname(
        path.dirname(path.dirname(path.dirname(path.dirname(__file__))))
    )
    print(f"project_root={project_root}")
    return project_root


def _construct_report(config, now, model, result_dict):

    report = {}
    report["name"] = config["name"]
    expt = config["experiment"]
    report["expriment"] = expt
    report["time_start"] = now.to_datetime_string()
    report = report | config["params"]["common"]
    report = report | config["params"]["validate"]

    hits, misses, ytest_ypred_dict = 0, 0, {}
    ground_truth_labels = []
    for k, v in result_dict.items():
        labelclass, predclass = k.split("-")[1], k.split("-")[3]
        if labelclass == predclass:
            hits = hits + v
        else:
            misses = misses + v
        ytest_ypred_dict[f"{labelclass}-{predclass}"] = v

        if not labelclass in ground_truth_labels:
            ground_truth_labels.append(labelclass)

    report["hits"] = hits
    report["misses"] = misses
    report["accuracy"] = hits / (hits + misses)

    perclass_accuracy = {}
    for tl in ground_truth_labels:
        chit, cmiss = 0, 0
        for tp in ground_truth_labels:
            tk = f"{tl}-{tp}"
            if tk in ytest_ypred_dict:
                if tl == tp:
                    chit = chit + ytest_ypred_dict[tk]
                else:
                    cmiss = cmiss + ytest_ypred_dict[tk]
        cacc = chit / (chit + cmiss)
        perclass_accuracy[f"{tl}_acc"] = cacc

    report["per_class_accuracy"] = dict(sorted(perclass_accuracy.items()))
    report["per_class_predictions"] = dict(sorted(ytest_ypred_dict.items()))

    return report


def main(experiment_timestamp, experiment_mode="model", experiment="baseline"):
    now = pendulum.now()
    torch.manual_seed(42)
    device = get_device()

    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")

    documents_fp = path.join(
        ASSETS_FP,
        "datasets",
        "named_entity_recognition",
        "model",
        f"test-{experiment}.csv",
    )
    vocabulary_fp = path.join(
        ASSETS_FP,
        "datasets",
        "named_entity_recognition",
        "model",
        f"vocabulary-{experiment}.csv",
    )
    labels_fp = path.join(
        ASSETS_FP, "datasets", "named_entity_recognition", "datamart", "labels.csv"
    )
    config_baseline_fp = path.join(
        ASSETS_FP, "config", "named_entity_recognition", f"config-{experiment}.yaml"
    )
    model_fp = path.join(
        ASSETS_FP,
        "models",
        "named_entity_recognition",
        f"{experiment_mode}-{experiment}-{experiment_timestamp}.pkl",
    )

    results_json_fp = path.join(
        ASSETS_FP,
        "models",
        "named_entity_recognition",
        f"report-{experiment}-{experiment_timestamp}.json",
    )

    # Corpus
    df = pd.read_csv(documents_fp, sep="\t")
    corpus = df["tokens"]
    # Labels
    labels_df = pd.read_csv(labels_fp, sep="\t")
    labels = df["onehot_labels"].to_list()
    count_unique_labels = labels_df.shape[0]
    # Config
    config = None
    with open(config_baseline_fp) as f:
        config = yaml.load(f)
    config_common = config.get("params").get("common")
    config_validate = config.get("params").get("validate")

    # Vocabulary
    word_to_idx_dict, vocab_size = load_vocabulary_to_idx(vocabulary_fp)

    # Data Loader
    word_idx_sequences, onehot_labels = transform_corpus_to_idx_word_windows(
        corpus, labels, word_to_idx_dict, pad_window_size=config_common["window_size"]
    )
    data = list(zip(word_idx_sequences, onehot_labels))
    collate_fn = partial(custom_collate_fn, device=device)
    loader = DataLoader(
        data,
        batch_size=config_validate["batch_size"],
        shuffle=config_validate["shuffle"],
        collate_fn=collate_fn,
    )
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

    result_dict = {}
    for test_instance, labels, _ in loader:
        output = model.forward(test_instance)
        for l_tsr, o_tsr in zip(labels, output):
            l = torch.argmax(l_tsr).to(torch.int32)
            o = torch.argmax(o_tsr).to(torch.int32)
            res = f"label-{l}-pred-{o}"
            if res in result_dict:
                c = result_dict[res]
                c = c + 1
                result_dict[res] = c
            else:
                result_dict[res] = 1

    j = _construct_report(config, now, model, result_dict)
    with open(results_json_fp, "w") as f:
        json.dump(j, f, ensure_ascii=False, indent=4)
    print(f"Done. Wrote results to {results_json_fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, required=True)
    args = parser.parse_args()
    main(experiment_timestamp=args.t)
