from typing import List, Set, Dict
import pandas as pd


def load_corpus(fp):
    df = pd.read_csv(fp, sep="\t")
    documents = df["document"].to_list()
    out_list = []
    for d in documents:
        l = d.split("^")
        out_list.append(l)
    return out_list


def load_labels(fp):
    df = pd.read_csv(fp, sep="\t")
    documents = df["labels"].to_list()
    out_list = []
    for d in documents:
        l = d.split("^")
        out_list.append([int(m) for m in l])
    return out_list


def save_vocabulary_to_idx(vocab_to_idx, fp):
    keys = [k for (k, v) in vocab_to_idx.items()]
    values = [v for (k, v) in vocab_to_idx.items()]
    df = pd.DataFrame({"k": keys, "v": values})
    df.to_csv(fp, sep="\t", index=False)
    return True


def load_vocabulary_to_idx(fp) -> Dict:
    df = pd.read_csv(fp, sep="\t")
    keys, values = df["k"].to_list(), df["v"].to_list()
    out_dict = {}
    for k, v in zip(keys, values):
        out_dict[k] = v
    vocab_size = len(out_dict)
    return out_dict, vocab_size
