from typing import List


def get_confusion_matrix(
    actual: List[int], predicted: List[int]
) -> List[List[int]]:
    fp, fn, tp, tn = 0, 0, 0, 0
    for act, pred in zip(actual, predicted):
        if act == pred:
            if pred == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred == 1:
                fp += 1
            else:
                fn += 1
    return [[tn,fp],[fn,tp]]


def accuracy(actual: List[int], predicted: List[int]) -> float:
    fp, fn, tp, tn = 0, 0, 0, 0
    for act, pred in zip(actual, predicted):
        if act == pred:
            if pred == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred == 1:
                fp += 1
            else:
                fn += 1
    return (tp+tn)/(tp+tn+fn+fp)


def precision(actual: List[int], predicted: List[int]) -> float:
    fp, fn, tp, tn = 0, 0, 0, 0
    for act, pred in zip(actual, predicted):
        if act == pred:
            if pred == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred == 1:
                fp += 1
            else:
                fn += 1
    return tp/(tp+fp)

def recall(actual: List[int], predicted: List[int]) -> float:
    fp, fn, tp, tn = 0, 0, 0, 0
    for act, pred in zip(actual, predicted):
        if act == pred:
            if pred == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred == 1:
                fp += 1
            else:
                fn += 1
    return tp/(tp+fn)


def f1(actual: List[int], predicted: List[int]) -> float:
    p = precision(actual, predicted)
    r = recall(actual, predicted)
    return (2*p*r)/(p+r)


def false_positive_rate(actual: List[int], predicted: List[int]) -> float:
    fp, fn, tp, tn = 0, 0, 0, 0
    for act, pred in zip(actual, predicted):
        if act == pred:
            if pred == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred == 1:
                fp += 1
            else:
                fn += 1
    return fp/(fp+tn)


def false_negative_rate(actual: List[int], predicted: List[int]) -> float:
    fp, fn, tp, tn = 0, 0, 0, 0
    for act, pred in zip(actual, predicted):
        if act == pred:
            if pred == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred == 1:
                fp += 1
            else:
                fn += 1
    return fn/(fn+tp)
