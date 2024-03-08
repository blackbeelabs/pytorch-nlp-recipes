from os import path

from utils.io_utils import *
from utils.text_utils import *


def _get_project_dir_folder():
    return path.dirname(path.dirname(path.dirname(path.dirname(__file__))))


def main():
    experiment = "baseline"
    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")

    for dataset in ["train", "test"]:
        documents_fp = path.join(ASSETS_FP, "datasets", "datamart", f"{dataset}.csv")
        df = pd.read_csv(documents_fp, sep="\t")
        corpus = df["tokens"].to_list()

        word_to_idx_dict = construct_word_to_idx(corpus)
        out_vocabulary_fp = path.join(
            ASSETS_FP, "datasets", "model", f"vocabulary-{experiment}.csv"
        )
        save_vocabulary_to_idx(word_to_idx_dict, out_vocabulary_fp)

        word_idx_sequences = transform_corpus_to_idx_sequences(corpus, word_to_idx_dict)
        df["token_sequences"] = word_idx_sequences
        df["token_sequences"] = df["token_sequences"].apply(
            lambda x: "^".join([str(i) for i in x])
        )
        baseline_fp = path.join(
            ASSETS_FP, "datasets", "model", f"{dataset}-{experiment}.csv"
        )
        df.to_csv(baseline_fp, sep="\t", index=False)
    print("Done")


if __name__ == "__main__":
    main()
