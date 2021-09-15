# Required packages:
from typing import Any, List, Tuple, Union
from numpy import ndarray
import pandas as pd
import re


def load_data(path: str) -> Tuple[List[str], List[str]]:
    """Loads data from file. Each except first (header) is a datapoint
    containing ID, Label, Email (content) separated by "\t".

    Args:
        path: Path to file from which to load data

    Returns:
        List of email contents and a list of lobels coresponding to each email.
    """
    df = pd.read_csv(path,sep='\t')
    emails = df['Email'].values[:].tolist()
    labelsString = df['Label'].values[:].tolist()
    labels = []
    for strs in labelsString:
        inp = ""
        if strs == 'spam':
            inp = 1
        elif strs == 'ham':
            inp = 0
        else:
            inp = ""
        labels.append(inp)

    return emails, labels


def preprocess(doc: str) -> str:
    """Preprocesses text to prepare it for feature extraction.

    Args:
        doc: String comprising the unprocessed contents of some email file.

    Returns:
        String comprising the corresponding preprocessed text.
    """
    # TODO
    ...
    splitted = re.split(r'[\.,:;?!]+',doc) #Regex for splitting on all the given delimiters
    # if "" in splitted: splitted.remove("")
    return ''.join(splitted)[:]


def preprocess_multiple(docs: List[str]) -> List[str]:
    """Preprocesses multiple texts to prepare them for feature extraction.

    Args:
        docs: List of strings, each consisting of the unprocessed contents
            of some email file.

    Returns:
        List of strings, each comprising the corresponding preprocessed
            text.
    """
    preprocessed = []
    for texts in docs:
        preprocessed.append(preprocess(texts))
    # TODO
    return preprocessed


def extract_features(
    train_dataset: List[str], test_dataset: List[str]
) -> Tuple[ndarray, ndarray]:
    """Extracts feature vectors from a preprocessed train and test datasets.

    Args:
        train_dataset: List of strings, each consisting of the preprocessed
            email content.
        test_dataset: List of strings, each consisting of the preprocessed
            email content.

    Returns:

    """
    # TODO
    print(train_dataset[0])
    return ["jaja"]


def train(X: ndarray, y: List[int]) -> object:
    """Trains a classifier on extracted feature vectors.

    Args:
        X: Numerical array-like object (2D) representing the instances.
        y: Numerical array-like object (1D) representing the labels.

    Returns:
        A trained model object capable of predicting over unseen sets of
            instances.
    """
    # TODO
    return None


def evaluate(
    y: List[int], y_pred: List[int]
) -> Tuple[float, float, float, float]:
    """Evaluates a model's predictive performance with respect to a labeled
    dataset.

    Args:
        y: Numerical array-like object (1D) representing the true labels.
        y_pred: Numerical array-like object (1D) representing the predicted
            labels.

    Returns:
        A tuple of four values: recall, precision, F_1, and accuracy.
    """
    # TODO
    return 0, 0, 0, 0


if __name__ == "__main__":
    print("Loading data...")
    train_data_raw, train_labels = load_data("data/train.tsv")
    test_data_raw, test_labels = load_data("data/test.tsv")

    print("Processing data...")
    t1 = preprocess(train_data_raw[0])
    print("T1:")
    print(t1)
    train_data = preprocess_multiple(train_data_raw)
    test_data = preprocess_multiple(test_data_raw)

    # print("Extracting features...")
    train_feature_vectors, test_feature_vectors = extract_features(
        train_data, test_data
    )

    # print("Training...")
    # classifier = train(train_feature_vectors, train_labels)

    # print("Applying model on test data...")
    # predicted_labels = classifier.predict(test_feature_vectors)

    # print("Evaluating")
    # recall, precision, f1, accuracy = evaluate(test_labels, predicted_labels)

    # print(f"Recall:\t{recall}")
    # print(f"Precision:\t{precision}")
    # print(f"F1:\t{f1}")
    # print(f"Accuracy:\t{accuracy}")
