from os import path
import re
import pandas as pd


def _get_project_dir_folder():
    project_root = path.dirname(
        path.dirname(path.dirname(path.dirname(path.dirname(__file__))))
    )
    print(f"project_root={project_root}")
    return project_root


def main():
    def _remove_emojis(s):
        emoj = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002500-\U00002BEF"  # chinese char
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u2066"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"  # dingbats
            "\u3030"
            "]+",
            re.UNICODE,
        )
        return re.sub(emoj, "", s)

    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")

    fp = path.join(
        ASSETS_FP, "datasets", "sentiment_analysis", "downloaded", "sentiment.csv"
    )
    df = pd.read_csv(fp)
    df = df[["Tweet Text", "Sentiment"]].copy()
    df.columns = ["text", "sentiment"]
    df["sentiment"] = df["sentiment"].map({"Negative": 0, "Positive": 1})

    df["text"] = df["text"].apply(_remove_emojis)
    df_train = df.sample(400)
    df_test = df[~df.index.isin(df_train.index)]

    df_train.to_csv(
        path.join(ASSETS_FP, "datasets", "sentiment_analysis", "datamart", "train.csv"),
        sep="\t",
        index=False,
    )
    df_test.to_csv(
        path.join(ASSETS_FP, "datasets", "sentiment_analysis", "datamart", "test.csv"),
        sep="\t",
        index=False,
    )
    print("Done")


if __name__ == "__main__":
    main()
