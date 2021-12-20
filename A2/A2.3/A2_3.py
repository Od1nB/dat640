from typing import Callable, Dict, List, Set

import ir_datasets
import csv


def load_rankings(
    filename: str = "system_rankings.tsv",
) -> Dict[str, List[str]]:
    """Load rankings from file. Every row in the file contains query ID and
    document ID separated by a tab ("\t").

        query_id    doc_id
        646	        4496d63c-8cf5-11e3-833c-33098f9e5267
        646	        ee82230c-f130-11e1-adc6-87dfa8eff430
        646	        ac6f0e3c-1e3c-11e3-94a2-6c66b668ea55

    Example return structure:

    {
        query_id_1: [doc_id_1, doc_id_2, ...],
        query_id_2: [doc_id_1, doc_id_2, ...]
    }

    Args:
        filename (optional): Path to file with rankings. Defaults to
            "system_rankings.tsv".

    Returns:
        Dictionary with query IDs as keys and list of documents as values.
    """
    return_dict = {}
    with open(filename) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            query_id, doc_id = row[0], row[1]
            if query_id == "query_id":
                continue
            if query_id in return_dict.keys():
                return_dict[query_id].append(doc_id)
            else:
                return_dict[query_id] = [doc_id]
    return return_dict


def load_ground_truth(
    collection: str = "wapo/v2/trec-core-2018",
) -> Dict[str, List[str]]:
    """Load ground truth from ir_datasets. Qrel is a namedtuple class with
    following properties:

        query_id: str
        doc_id: str
        relevance: int
        iteration: str

    relevance is split into levels with values:

        0	not relevant
        1	relevant
        2	highly relevant

    This function considers documents to be relevant for relevance values
        1 and 2.

    Generic structure of returned dictionary:

    {
        query_id_1: {doc_id_1, doc_id_3, ...},
        query_id_2: {doc_id_1, doc_id_5, ...}
    }

    Args:
        filename (optional): Path to file with rankings. Defaults to
            "system_rankings.tsv".

    Returns:
        Dictionary with query IDs as keys and sets of documents as values.
    """
    dataset = ir_datasets.load(collection)
    return_dict = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance == 1 or qrel.relevance == 2:
            if qrel.query_id in return_dict.keys():
                if qrel.doc_id in return_dict[qrel.query_id]:
                    continue
                return_dict[qrel.query_id].append(qrel.doc_id)
            else:
                return_dict[qrel.query_id] = [qrel.doc_id]
    return return_dict


def get_precision(
    system_ranking: List[str], ground_truth: Set[str], k: int = 100
) -> float:
    """Computes Precision@k.

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.
        k: Cutoff. Only consider system rankings up to k.

    Returns:
        P@K (float).
    """
    reccomended = system_ranking[:k]
    rel = 0
    for doc in reccomended:
        if doc in ground_truth:
            rel+= 1
    return rel/k


def get_average_precision(
    system_ranking: List[str], ground_truth: Set[str]
) -> float:
    """Computes Average Precision (AP).

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.

    Returns:
        AP (float).
    """
    rel = 0
    precisions = []
    for ind in range(0,len(system_ranking)):
        if system_ranking[ind] in ground_truth:
            rel += 1
            precisions.append(rel/(ind+1))
    return sum(precisions)/len(ground_truth)


def get_reciprocal_rank(
    system_ranking: List[str], ground_truth: Set[str]
) -> float:
    """Computes Reciprocal Rank (RR).

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.

    Returns:
        RR (float).
    """
    for i in range(0,len(system_ranking)):
        if system_ranking[i] in ground_truth:
            return 1/(i+1)
    return 0


def get_mean_eval_measure(
    system_rankings: Dict[str, List[str]],
    ground_truths: Dict[str, Set[str]],
    eval_function: Callable,
) -> float:
    """Computes a mean of any evaluation measure over a set of queries.

    Args:
        system_rankings: Dict with query ID as key and a ranked list of
            document IDs as value.
        ground_truths: Dict with query ID as key and a set of relevant document
            IDs as value.
        eval_function: Callback function for the evaluation measure that mean
            is computed over.

    Returns:
        Mean evaluation measure (float).
    """
    querynumbs = []
    for query in system_rankings.keys():
        querynumbs.append(eval_function(system_rankings[query], ground_truths[query]))
    return sum(querynumbs)/len(system_rankings)


if __name__ == "__main__":
    system_rankings = load_rankings()
    ground_truths = load_ground_truth()
    print(ground_truths)