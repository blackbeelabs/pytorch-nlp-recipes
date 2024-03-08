from os import path
import csv
import pandas as pd


def _get_project_dir_folder():
    return path.dirname(path.dirname(path.dirname(path.dirname(__file__))))


def _to_onehot_labels(label_seq, labels_dict):
    ls = label_seq.split("^")
    onehot_lbls = []
    for l in ls:
        onehot = [0] * len(labels_dict)
        onehot[labels_dict[l]] = 1
        onehot_lbls.append("".join([str(i) for i in onehot]))
    return "^".join(onehot_lbls)


def main():
    # 1 - to datamart format
    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")
    for dataset in ["train", "test"]:

        in_fp = path.join(ASSETS_FP, "datasets", "downloaded", f"{dataset}.txt")
        out_fp = path.join(ASSETS_FP, "datasets", "datamart", f"{dataset}.csv")

        tokens_list = []
        with open(in_fp, newline="") as csvfile:
            spamreader = csv.reader(csvfile, delimiter="^", quotechar="|")
            for row in spamreader:
                tokens_list.append(row)

        sentences_list = []
        sentence = []
        for t in tokens_list[1:]:
            if len(t) == 0:
                if len(sentence) == 0:
                    pass
                else:
                    sentences_list.append(sentence)
                    sentence = []
            else:
                sentence.append(t)

        token_sequences = []
        label_sequences = []
        for s in sentences_list:
            tokens, labels = [], []
            for t in s:
                w = t[0].split(" ")[0]
                l = t[0].split(" ")[-1]
                if l.startswith("I-"):
                    l = l[2:]
                tokens.append(w)
                labels.append(l)
            token_sequences.append(tokens)
            label_sequences.append(labels)

        t_col, l_col = [], []
        for t, l in zip(token_sequences, label_sequences):
            t_col.append("^".join(t))
            l_col.append("^".join(l))
        df_out = pd.DataFrame({"tokens": t_col, "labels": l_col})
        df_out[["tokens", "labels"]].to_csv(out_fp, sep="\t", index=False)

    # 2 - unique labels
    datamart_fp = path.join(ASSETS_FP, "datasets", "datamart", "train.csv")
    uniq_labels_fp = path.join(ASSETS_FP, "datasets", "datamart", "labels.csv")
    df = pd.read_csv(datamart_fp, sep="\t")
    labels = df["labels"]
    unique_labels = []
    for l in labels:
        llist = l.split("^")
        unique_labels.extend(llist)
    unique_labels = list(set(unique_labels))
    unique_labels_df = pd.DataFrame(
        {
            "v": [i for i, _ in enumerate(unique_labels)],
            "label": [l for _, l in enumerate(unique_labels)],
        }
    )
    unique_labels_df.to_csv(
        uniq_labels_fp,
        sep="\t",
        index=False,
    )

    # 3 - add labels in onehot format
    unique_labels_dict = {}
    for _, r in unique_labels_df.iterrows():
        unique_labels_dict[r["label"]] = r["v"]

    for dataset in ["train", "test"]:
        datamart_fp = path.join(ASSETS_FP, "datasets", "datamart", f"{dataset}.csv")
        datamart_df = pd.read_csv(datamart_fp, sep="\t")
        datamart_df["onehot_labels"] = datamart_df["labels"].apply(
            _to_onehot_labels, labels_dict=unique_labels_dict
        )
        datamart_df.to_csv(datamart_fp, sep="\t", index=False)
    print("Done")


if __name__ == "__main__":
    main()
