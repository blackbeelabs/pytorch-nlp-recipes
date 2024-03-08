from os import path
import re
import pandas as pd


def _get_project_dir_folder():
    return path.dirname(path.dirname(path.dirname(path.dirname(__file__))))


def main():

    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")

    fp = path.join(ASSETS_FP, "datasets", "downloaded", "sentiment.csv")
    df = pd.read_csv(fp)
    df = df[["Tweet Text", "Sentiment"]].copy()
    df.columns = ["text", "sentiment"]
    df["sentiment"] = df["sentiment"].map({"Negative": 0, "Positive": 1})

    df["text"] = df["text"].apply(_remove_emojis)
    df_train = df.sample(400)
    df_test = df[~df.index.isin(df_train.index)]

    df_train.to_csv(
        path.join(ASSETS_FP, "datasets", "datamart", "train.csv"),
        sep="\t",
        index=False,
    )
    df_test.to_csv(
        path.join(ASSETS_FP, "datasets", "datamart", "test.csv"),
        sep="\t",
        index=False,
    )
    print("Done")


if __name__ == "__main__":
    main()
