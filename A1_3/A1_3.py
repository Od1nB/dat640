# Required packages:
from typing import Any, List, Tuple, Union
from numpy import ndarray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import pandas as pd
import re

tfidVector = TfidfVectorizer()

def load_data(path: str) -> Tuple[List[str], List[str]]:
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
            inp = 1
        labels.append(inp)

    return emails, labels


def preprocess(doc: str) -> str:
    splitted = re.split(r'[\.,:;?!]+',doc) #Regex for splitting on all the given delimiters
    return ''.join(splitted)[:]


def preprocess_multiple(docs: List[str]) -> List[str]:
    preprocessed = []
    for texts in docs:
        preprocessed.append(preprocess(texts))
    return preprocessed


def extract_features(
    train_dataset: List[str], test_dataset: List[str]
) -> Tuple[ndarray, ndarray]:
    X_train = tfidVector.fit_transform(train_dataset)
    Y_test = tfidVector.transform(test_dataset)
    return (X_train, Y_test)

def train(X: ndarray, y: List[int]) -> object:
    clf = SGDClassifier(max_iter=1000, tol=1e-3)
    clf.fit(X,y)
    return clf


def evaluate(
    y: List[int], y_pred: List[int]
) -> Tuple[float, float, float, float]:
    #Using metrics built in score methods
    return metrics.recall_score(y, y_pred), metrics.precision_score(y,y_pred), metrics.f1_score(y, y_pred), metrics.accuracy_score(y, y_pred)


if __name__ == "__main__":
    print("Loading data...")
    train_data_raw, train_labels = load_data("data/train.tsv")
    test_data_raw, test_labels = load_data("data/test.tsv")

    print("Processing data...")
    train_data = preprocess_multiple(train_data_raw)
    test_data = preprocess_multiple(test_data_raw)

    # print("Extracting features...")
    train_feature_vectors, test_feature_vectors = extract_features(
        train_data, test_data
    )

    print("Training...")
    classifier = train(train_feature_vectors, train_labels)

    print("Applying model on test data...")
    predicted_labels = classifier.predict(test_feature_vectors)

    print("Evaluating")
    recall, precision, f1, accuracy = evaluate(test_labels, predicted_labels)

    print(f"Recall:\t{recall}")
    print(f"Precision:\t{precision}")
    print(f"F1:\t{f1}")
    print(f"Accuracy:\t{accuracy}")
