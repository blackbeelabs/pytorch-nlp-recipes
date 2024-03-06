import json
from os import path
from functools import partial

import torch
from torch.utils.data import DataLoader
from ruamel.yaml import YAML
from tqdm import tqdm
import numpy as np

from utils.io_utils import *
from utils.text_utils import *
from model.model import *
from tasks.tasks import *

yaml = YAML()


def _get_project_dir_folder():
    return path.dirname(path.dirname(path.dirname(path.dirname(__file__))))


def _construct_report(config, experiment_timestamp, result_dict):
    report = {}
    report["name"] = config["name"]
    expt = config["experiment"]
    report["time_start"] = experiment_timestamp
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


def main():

    torch.manual_seed(42)
    device = get_device()

    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")

    workflow = "model"
    experiment = "baseline"
    experiment_timestamp = "20240306T114509"

    documents_fp = path.join(ASSETS_FP, "datasets", "datamart", f"test.csv")
    vocabulary_fp = path.join(
        ASSETS_FP,
        "datasets",
        "model",
        f"vocabulary-{workflow}-{experiment}-{experiment_timestamp}.csv",
    )
    config_fp = path.join(ASSETS_FP, "config", f"config-{experiment}.yaml")
    model_fp = path.join(
        ASSETS_FP,
        "models",
        f"{workflow}-{experiment}-{experiment_timestamp}.pkl",
    )

    results_json_fp = path.join(
        ASSETS_FP,
        "models",
        f"report-{workflow}-{experiment}-{experiment_timestamp}.json",
    )
    results_csv_fp = path.join(
        ASSETS_FP,
        "models",
        f"detailedreport-{workflow}-{experiment}-{experiment_timestamp}.csv",
    )

    # Corpus
    corpus = get_corpus_task(documents_fp)
    # Labels
    labels = get_labels_task(documents_fp)
    # Config
    config = get_config_task(config_fp)
    config_common = config.get("params").get("common")
    config_validate = config.get("params").get("validate")
    # Vocabulary
    word_to_idx_dict, vocab_size = load_vocabulary_to_idx(vocabulary_fp)
    # Data Loader
    word_idx_sequences = transform_corpus_to_idx_sequences(
        corpus,
        word_to_idx_dict,
    )
    # Data loader
    data = list(zip(word_idx_sequences, labels))
    collate_fn = partial(custom_collate_fn, word_to_idx=word_to_idx_dict, device=device)
    loader = DataLoader(
        data,
        batch_size=config_validate["batch_size"],
        shuffle=config_validate["shuffle"],
        collate_fn=collate_fn,
    )
    # Load model
    model_hyperparameters = get_model_hyperparameters_task(
        config_common, config_validate
    )
    model = SentimentClassifierBaseline(
        model_hyperparameters,
        vocab_size,
    )
    model.load_state_dict(torch.load(model_fp))

    # Evaluate
    result_dict, result_list = {}, []
    r = 0
    for test_sequences_BL, labels_BS in tqdm(loader):
        # Outputs from model
        outputs = model(test_sequences_BL).squeeze(1)
        outputs = outputs.detach().numpy()
        to_classes = lambda x: 1 if x >= 0.5 else 0
        to_classes_v = np.vectorize(to_classes)
        outputs = to_classes_v(outputs)
        # Outputs from label
        labels = labels_BS.squeeze(1).numpy()
        labels = labels.astype(int)

        for l, o in zip(labels, outputs):
            res = f"label-{l}-pred-{o}"
            if res in result_dict:
                c = result_dict[res]
                c = c + 1
                result_dict[res] = c
            else:
                result_dict[res] = 1
            result_list.append([r, l, o])
            r += 1

    j = _construct_report(config, experiment_timestamp, result_dict)

    with open(results_json_fp, "w") as f:
        json.dump(j, f, ensure_ascii=False, indent=4)
    print(f"Done. Wrote results to {results_json_fp}")

    results_df = pd.DataFrame(
        result_list, columns=["test_example_index", "ytest", "ypredict"]
    )
    results_df.to_csv(results_csv_fp, index=False)


if __name__ == "__main__":
    main()
