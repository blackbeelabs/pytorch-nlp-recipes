import re


def remove_emojis(s):
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


def to_labels(label_seq, labels_dict):
    ls = label_seq.split("^")
    onehot_lbls = []
    for l in ls:
        onehot = [0] * len(labels_dict)
        onehot[labels_dict[l]] = 1
        onehot_lbls.append("".join([str(i) for i in onehot]))
    return "^".join(onehot_lbls)


def construct_word_to_idx(corpus):
    def _construct_vocabulary(corpus):
        # Find all the unique words in our corpus
        vocabulary = list(set(w for s in corpus for w in s.split()))
        vocabulary = sorted(vocabulary)
        # Add the pad and unknown token to our vocabulary
        vocabulary = ["<pad>", "<unk>"] + vocabulary
        return vocabulary

    idx_to_word = _construct_vocabulary(corpus)

    # Creating a dictionary to find the index of a given word
    word_to_idx = {word: ind for ind, word in enumerate(idx_to_word)}
    vocab_size = len(word_to_idx)
    return word_to_idx, vocab_size


def transform_corpus_to_idx_sequences(corpus, word_to_idx):
    def _convert_token_to_indices(sentence, word_to_ix):
        return [
            word_to_ix.get(token, word_to_ix["<unk>"]) for token in sentence.split()
        ]

    indices = [
        _convert_token_to_indices(
            d,
            word_to_idx,
        )
        for d in corpus
    ]
    return indices
