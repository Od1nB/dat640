import json
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
from elasticsearch import Elasticsearch

FIELDS = ["title", "body"]

INDEX_SETTINGS = {
    "properties": {
        "title": {"type": "text", "term_vector": "yes", "analyzer": "english"},
        "body": {"type": "text", "term_vector": "yes", "analyzer": "english"},
    }
}

FEATURES_QUERY = [
    "query_length",
    "query_sum_idf",
    "query_max_idf",
    "query_avg_idf",
]
FEATURES_DOC = ["doc_length_title", "doc_length_body"]
FEATURES_QUERY_DOC = [
    "unique_query_terms_in_title",
    "sum_TF_title",
    "max_TF_title",
    "avg_TF_title",
    "unique_query_terms_in_body",
    "sum_TF_body",
    "max_TF_body",
    "avg_TF_body",
]


def analyze_query(
    es: Elasticsearch, query: str, field: str, index: str = "toy_index"
) -> List[str]:
    """Analyzes a query with respect to the relevant index.

    Args:
        es: Elasticsearch object instance.
        query: String of query terms.
        field: The field with respect to which the query is analyzed.
        index: Name of the index with respect to which the query is analyzed.

    Returns:
        A list of query terms that exist in the specified field among the
        documents in the index.
    """
    tokens = es.indices.analyze(index=index, body={"text": query})["tokens"]
    query_terms = []
    for t in sorted(tokens, key=lambda x: x["position"]):
        # Use a boolean query to find at least one document that contains the
        # term.
        hits = (
            es.search(
                index=index,
                query={"match": {field: t["token"]}},
                _source=False,
                size=1,
            )
            .get("hits", {})
            .get("hits", {})
        )
        doc_id = hits[0]["_id"] if len(hits) > 0 else None
        if doc_id is None:
            continue
        query_terms.append(t["token"])
    return query_terms


def get_doc_term_freqs(
    es: Elasticsearch, doc_id: str, field: str, index: str = "toy_index"
) -> Dict[str, int]:
    """Gets the term frequencies of a field of an indexed document.

    Args:
        es: Elasticsearch object instance.
        doc_id: Document identifier with which the document is indexed.
        field: Field of document to consider for term frequencies.
        index: Name of the index where document is indexed.

    Returns:
        Dictionary of terms and their respective term frequencies in the field
        and document.
    """
    tv = es.termvectors(
        index=index, id=doc_id, fields=field, term_statistics=True
    )
    if tv["_id"] != doc_id:
        return None
    if field not in tv["term_vectors"]:
        return None
    term_freqs = {}
    for term, term_stat in tv["term_vectors"][field]["terms"].items():
        term_freqs[term] = term_stat["term_freq"]
    return term_freqs


def extract_query_features(
    query_terms: List[str], es: Elasticsearch, index: str = "toy_index"
) -> Dict[str, float]:
    """Extracts features of a query.

        Args:
            query_terms: List of analyzed query terms.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service.
        Returns:
            Dictionary with keys 'query_length', 'query_sum_idf',
                'query_max_idf', and 'query_avg_idf'.
    """
    # TODO
    return {}


def extract_doc_features(
    doc_id: str, es: Elasticsearch, index: str = "toy_index"
) -> Dict[str, float]:
    """Extracts features of a document.

        Args:
            doc_id: Document identifier of indexed document.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service.

        Returns:
            Dictionary with keys 'doc_length_title', 'doc_length_body'.
    """
    # TODO
    return {}


def extract_query_doc_features(
    query_terms: List[str],
    doc_id: str,
    es: Elasticsearch,
    index: str = "toy_index",
) -> Dict[str, float]:
    """Extracts features of a query and document pair.

        Args:
            query_terms: List of analyzed query terms.
            doc_id: Document identifier of indexed document.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service.

        Returns:
            Dictionary with keys 'unique_query_terms_in_title',
                'unique_query_terms_in_body', 'sum_TF_title', 'sum_TF_body',
                'max_TF_title', 'max_TF_body', 'avg_TF_title', 'avg_TF_body'. 
    """
    # TODO
    return {}


def extract_features(
    query_terms: List[str],
    doc_id: str,
    es: Elasticsearch,
    index: str = "toy_index",
) -> List[float]:
    """Extracts query features, document features and query-document features
        of a query and document pair.

        Args:
            query_terms: List of analyzed query terms.
            doc_id: Document identifier of indexed document.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service.

        Returns:
            List of extracted feature values in a fixed order.
    """
    query_features = extract_query_features(query_terms, es, index=index)
    feature_vect = [query_features[f] for f in FEATURES_QUERY]

    doc_features = extract_doc_features(doc_id, es, index=index)
    feature_vect.extend([doc_features[f] for f in FEATURES_DOC])

    query_doc_features = extract_query_doc_features(
        query_terms, doc_id, es, index=index
    )
    feature_vect.extend([query_doc_features[f] for f in FEATURES_QUERY_DOC])

    return feature_vect


def index_documents(filepath: str, es: Elasticsearch, index: str) -> None:
    """Indexes documents from JSONL file."""
    bulk_data = []
    with open(filepath, "r") as docs:
        for doc in docs:
            doc = json.loads(doc)
            bulk_data.append(
                {"index": {"_index": index, "_id": doc.pop("doc_id")}}
            )
            bulk_data.append(doc)
    es.bulk(index=index, body=bulk_data, refresh=True)


def load_queries(filepath: str) -> Dict[str, str]:
    """Given a filepath, returns a dictionary with query IDs and corresponding
    query strings.

    This is an example query:

    ```
    <top>
    <num> Number: OHSU1
    <title> 60 year old menopausal woman without hormone replacement therapy
    <desc> Description:
    Are there adverse effects on lipids when progesterone is given with estrogen replacement therapy
    </top>

    ```

    Take as query ID the value (on the same line) after `<num> Number: `, 
    and take as the query string the rest of the line after `<title> `. Omit
    newline characters.

    Args:
        filepath: String (constructed using os.path) of the filepath to a
        file with queries.

    Returns:
        A dictionary with query IDs and corresponding query strings.
    """
    q_dict = {}
    with open(filepath) as f:
        lines = f.readlines()
    q_id = ""
    for line in lines:
        if "<num> Number:" in line:
            q_id = line.replace("<num> Number:","").replace("\n","")
        if "<title>" in line:
            s = line.replace("<title> ","")
            if s:
                q_dict[q_id] = s
    return q_dict


def load_qrels(filepath: str) -> Dict[str, List[str]]:
    """Loads query relevance judgments from a file.
    The qrels file has content with tab-separated values such as the following:

    ```
    MSH1	87056458
    MSH1	87056800
    MSH1	87058606
    MSH2	87049102
    MSH2	87056792
    ```

    Args:
        filepath: String (constructed using os.path) of the filepath to a
            file with queries.

    Returns:
        A dictionary with query IDs and a corresponding list of document IDs
            for documents judged relevant to the query.
    """
    # TODO
    qrels_dict = {}
    with open(filepath) as f:
        lines = f.readlines()
    for line in lines:
        query, doc_id = line.split(maxsplit=1)
        if query in qrels_dict.keys():
            qrels_dict[query] = qrels_dict[query] + [doc_id]
        else:
            qrels_dict[query] = [doc_id]


    return qrels_dict


def prepare_ltr_training_data(
    query_ids: List[str],
    all_queries: Dict[str, str],
    all_qrels: Dict[str, List[str]],
    es: Elasticsearch,
    index: str,
) -> Tuple[List[List[float]], List[int]]:
    """Prepares feature vectors and labels for query and document pairs found
    in the training data.

        Args:
            query_ids: List of query IDs.
            all_queries: Dictionary containing all queries.
            all_qrels: Dictionary with keys as query ID and values as list of
                relevant documents.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service.

        Returns:
            X: List of feature vectors extracted for each pair of query and
                retrieved or relevant document.
            y: List of corresponding labels.
    """
    # TODO
    return [], []


class PointWiseLTRModel:
    def __init__(self) -> None:
        """Instantiates LTR model with an instance of scikit-learn regressor.
        """
        # TODO
        self.regressor = ...

    def _train(self, X: List[List[float]], y: List[float]) -> None:
        """Trains an LTR model.

        Args:
            X: Features of training instances.
            y: Relevance assessments of training instances.
        """
        assert self.regressor is not None
        self.model = self.regressor.fit(X, y)

    def rank(
        self, ft: List[List[str]], doc_ids: List[str]
    ) -> List[Tuple[str, int]]:
        """Predicts relevance labels and rank documents for a given query.

        Args:
            ft: A list of feature vectors for query-document pairs.
            doc_ids: A list of document ids.
        Returns:
            List of tuples, each consisting of document ID and predicted
                relevance label.
        """
        assert self.model is not None
        rel_labels = self.model.predict(ft)
        sort_indices = np.argsort(rel_labels)[::-1]

        results = []
        for i in sort_indices:
            results.append((doc_ids[i], rel_labels[i]))
        return results


def get_rankings(
    ltr: PointWiseLTRModel,
    query_ids: List[str],
    all_queries: Dict[str, str],
    es: Elasticsearch,
    index: str,
    rerank: bool = False,
) -> Dict[str, List[str]]:
    """Generate rankings for each of the test query IDs.

    Args:
        ltr: A trained PointWiseLTRModel instance.
        query_ids: List of query IDs.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.
        rerank: Boolean flag indicating whether the first-pass retrieval
            results should be reranked using the LTR model.

    Returns:
        A dictionary of rankings for each test query ID.
    """

    test_rankings = {}
    for i, query_id in enumerate(query_ids):
        print(
            "Processing query {}/{} ID {}".format(
                i + 1, len(query_ids), query_id
            )
        )
        # First-pass retrieval
        query_terms = analyze_query(
            es, all_queries[query_id], "body", index=index
        )
        if len(query_terms) == 0:
            print(
                "WARNING: query {} is empty after analysis; ignoring".format(
                    query_id
                )
            )
            continue
        hits = es.search(
            index=index, q=" ".join(query_terms), _source=True, size=100
        )["hits"]["hits"]
        test_rankings[query_id] = [hit["_id"] for hit in hits]

        # Rerank the first-pass result set using the LTR model.
        if rerank:
            # TODO
            ...
    return test_rankings


def get_reciprocal_rank(
    system_ranking: List[str], ground_truth: List[str]
) -> float:
    """Computes Reciprocal Rank (RR).

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.

    Returns:
        RR (float).
    """
    for i, doc_id in enumerate(system_ranking):
        if doc_id in ground_truth:
            return 1 / (i + 1)
    return 0


def get_mean_eval_measure(
    system_rankings: Dict[str, List[str]],
    ground_truths: Dict[str, Set[str]],
    eval_function: Callable,
) -> float:
    """Computes a mean of any evaluation measure over a set of queries.

    Args:
        system_rankings: Dict with query ID as key and a ranked list of document
            IDs as value.
        ground_truths: Dict with query ID as key and a set of relevant document
            IDs as value.
        eval_function: Callback function for the evaluation measure that mean is
            computed over.

    Returns:
        Mean evaluation measure (float).
    """
    sum_score = 0
    for query_id, system_ranking in system_rankings.items():
        sum_score += eval_function(system_ranking, ground_truths[query_id])
    return sum_score / len(system_rankings)


if __name__ == "__main__":
    index_name = "trec9_index"
    es = Elasticsearch(timeout=120)
    qs = load_qrels("./data/qrels")
    # index_documents("data/documents.jsonl", es, index_name)
