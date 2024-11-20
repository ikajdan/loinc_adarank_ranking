import os
import re
import string as st
import sys

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import tabulate
from nltk import PorterStemmer, WordNetLemmatizer
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.abspath("adarank_lib"))
# pylint: disable=wrong-import-position
from adarank_lib.adarank import AdaRank
from adarank_lib.metrics import NDCGScorer
from adarank_lib.utils import group_offsets, load_docno

DATA_FILE = "loinc_dataset.xlsx"
OUTPUT_DIR = "out"
RANKING_FILE = os.path.join(OUTPUT_DIR, "ranking.txt")


def download_nltk_resources():
    nltk_data_dir = os.path.join(os.getcwd(), ".nltk_data")
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
        nltk.download("wordnet", download_dir=nltk_data_dir)
        nltk.download("stopwords", download_dir=nltk_data_dir)
    nltk.data.path.append(nltk_data_dir)


def preprocess_text(text):
    ps = PorterStemmer()
    stopwords = set(nltk.corpus.stopwords.words("english"))
    word_net = WordNetLemmatizer()

    text = "".join([ch for ch in text if ch not in st.punctuation])
    tokens = [x.lower() for x in re.split(r"\s+", text)]
    tokens = [x for x in tokens if len(x) > 3]
    tokens = [word for word in tokens if word not in stopwords]
    tokens = [ps.stem(word) for word in tokens]
    tokens = [word_net.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def excel_to_df(excel_file):
    sheets_dict = pd.read_excel(excel_file, sheet_name=None)
    all_sheets = [sheet for _, sheet in sheets_dict.items()]
    df = pd.concat(all_sheets)
    df.reset_index(inplace=True, drop=True)
    return df


def get_train_test_df(df):
    df = df.sample(frac=1, random_state=2)
    train_size = int(0.7 * len(df))
    train_set = df[:train_size].sort_values(by=["qid"])
    test_set = df[train_size:].sort_values(by=["qid"])
    return train_set, test_set


def df_to_svmlight_files(df, output_dir):
    main_file = os.path.join(output_dir, "main_no_ids.dat")
    train_file = os.path.join(output_dir, "train_no_ids.dat")
    test_file = os.path.join(output_dir, "test_no_ids.dat")

    tfidf = TfidfVectorizer()
    df["clean_text"] = df["long_common_name"].apply(preprocess_text)
    x = tfidf.fit_transform(df["clean_text"]).toarray()

    df["features"] = x.tolist()
    train_df, test_df = get_train_test_df(df)

    qid = df["qid"].to_numpy()
    y = df["relevance"].to_numpy() + 1
    x_train = np.array(train_df["features"].values.tolist())
    qid_train = train_df["qid"].to_numpy()
    y_train = train_df["relevance"].to_numpy() + 1
    x_test = np.array(test_df["features"].values.tolist())
    qid_test = test_df["qid"].to_numpy()
    y_test = test_df["relevance"].to_numpy() + 1

    dump_svmlight_file(x, y, main_file, query_id=qid)
    dump_svmlight_file(x_train, y_train, train_file, query_id=qid_train)
    dump_svmlight_file(x_test, y_test, test_file, query_id=qid_test)

    out_file, out_train_file, out_test_file = add_ids(
        df["id"].tolist(),
        main_file,
        train_df["id"].tolist(),
        train_file,
        test_df["id"].tolist(),
        test_file,
        output_dir,
    )

    for fname in os.listdir(output_dir):
        if "no_ids" in fname:
            os.remove(os.path.join(output_dir, fname))

    return out_file, out_train_file, out_test_file


def add_ids(ids, file, train_ids, train_file, test_ids, test_file, output_dir):
    output_files = {
        "main": (file, ids, os.path.join(output_dir, "main.dat")),
        "train": (
            train_file,
            train_ids,
            os.path.join(output_dir, "train.dat"),
        ),
        "test": (
            test_file,
            test_ids,
            os.path.join(output_dir, "test.dat"),
        ),
    }

    for _, (input_file, ids_list, output_file) in output_files.items():
        with open(input_file, "r", encoding="utf-8") as fin, open(
            output_file, "w", encoding="utf-8"
        ) as fout:
            for index, line in enumerate(fin):
                docno = ids_list[index] if index < len(ids_list) else ""
                fout.write(f"{line.strip()} #{docno}\n")

    return tuple(file_info[2] for file_info in output_files.values())


def plot_ndcg_scores(y_test, pred, qid_test, k_values=(1, 2, 3, 4, 5, 10, 20)):
    scores = {k: NDCGScorer(k=k)(y_test, pred, qid_test).mean() for k in k_values}

    df = pd.DataFrame(list(scores.items()), columns=["k", "NDCG Score"])

    sns.lineplot(data=df, x="k", y="NDCG Score", marker="o", color="b")

    plt.xlabel("k")
    plt.ylabel("NDCG Score")
    plt.title("NDCG Scores vs Cutoff")

    plt.grid(True)
    plt.show()


def print_ranking(qid, docno, pred, output=None):
    table = []
    headers = ["qid", "docno", "rank", "score"]
    if output is None:
        output = sys.stdout
    for a, b in group_offsets(qid):
        idx = np.argsort(-pred[a:b]) + a
        for rank, i in enumerate(idx, 1):
            table.append([qid[i], docno[i], rank, round(pred[i], 3)])
    output.write(tabulate.tabulate(table, headers, tablefmt="pretty"))


def main():
    # pylint: disable=unbalanced-tuple-unpacking
    download_nltk_resources()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = excel_to_df(DATA_FILE)

    _, train_file, test_file = df_to_svmlight_files(df, OUTPUT_DIR)

    x_train, y_train, qid_train = load_svmlight_file(train_file, query_id=True)
    x_test, y_test, qid_test = load_svmlight_file(test_file, query_id=True)

    model = AdaRank(max_iter=100, estop=10, scorer=NDCGScorer(k=10)).fit(
        x_train, y_train, qid_train
    )
    pred = model.predict(x_test, qid_test)

    plot_ndcg_scores(y_test, pred, qid_test)

    docno = load_docno(test_file, letor=False)
    with open(RANKING_FILE, "w", encoding="utf-8") as output:
        print_ranking(qid_test, docno, pred, output)


if __name__ == "__main__":
    main()
