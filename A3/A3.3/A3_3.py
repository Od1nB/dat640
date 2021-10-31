import abc
from collections import UserDict as DictClass
from typing import Dict, List
import math

CollectionType = Dict[str, Dict[str, List[str]]]


class DocumentCollection(DictClass):
    """Document dictionary class with helper functions."""

    def total_field_length(self, field: str) -> int:
        """Total number of terms in a field for all documents."""
        return sum(len(fields[field]) for fields in self.values())

    def avg_field_length(self, field: str) -> float:
        """Average number of terms in a field across all documents."""
        return self.total_field_length(field) / len(self)

    def get_field_documents(self, field: str) -> Dict[str, List[str]]:
        """Dictionary of documents for a single field."""
        return {
            doc_id: doc[field] for (doc_id, doc) in self.items() if field in doc
        }


class Scorer(abc.ABC):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        feature_weights=[0.85, 0.1, 0.05],
        mu: float = 100,
        window: int = 3,
    ):
        """Interface for the scorer class.

        Args:
            collection: Collection of documents. Needed to calculate document
                statistical information.
            index: Index to use for calculating scores.
            feature_weights: Weights associated with each feature function
            mu: Smoothing parameter
            window: Window for unordered feature function.
        """
        self.collection = collection
        self.index = index

        if not sum(feature_weights) == 1:
            raise ValueError("Feature weights should sum to 1.")

        self.feature_weights = feature_weights
        self.mu = mu
        self.window = window

    def score_collection(self, query_terms: List[str]):
        """Scores all documents in the collection using document-at-a-time query
        processing.

        Params:
            query_term: Sequence (list) of query terms.

        Returns:
            Dict with doc_ids as keys and retrieval scores as values.
            (It may be assumed that documents that are not present in this dict
            have a retrival score of 0.)
        """
        lT, lO, lU = self.feature_weights
        return {
            doc_id: (
                lT * self.unigram_matches(query_terms, doc_id)
                + lO * self.ordered_bigram_matches(query_terms, doc_id)
                + lU * self.unordered_bigram_matches(query_terms, doc_id)
            )
            for doc_id in self.collection
        }

    @abc.abstractmethod
    def unigram_matches(self, query_terms: List[str], doc_id: str) -> float:
        """Returns unigram matches based on smoothed entity language model.

        Args:
            query_terms: List of query terms
            doc_id: Document ID for the document we wish to score

        Returns:
            Score for unigram matches for document with doc ID.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ordered_bigram_matches(self, query_terms: List[str], doc_id):
        """Returns ordered bigram matches based on smoothed entity language
        model.

        Args:
            query_terms: List of query terms
            doc_id: Document ID for the document we wish to score

        Returns:
            Score for ordered bigram matches for document with doc ID.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def unordered_bigram_matches(self, query_terms: List[str], doc_id):
        """Returns unordered bigram matches based on smoothed entity language
        model.

        Args:
            query_terms: List of query terms
            doc_id: Document ID for the document we wish to score

        Returns:
            Score for unordered bigram matches for document with doc ID.
        """
        raise NotImplementedError


class SDMScorer(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        feature_weights=[0.85, 0.1, 0.05],
        mu: float = 100,
        window: int = 3,
    ):
        """SDM scorer. This scorer can be applied on a single field only.

        Args:
            collection: Collection of documents. Needed to calculate document
                statistical information.
            index: Index to use for calculating scores.
            field (optional): Single field to use in scoring.. Defaults to None.
            fields (optional): List of fields to use in scoring. Defaults to
                None.
        """
        super().__init__(collection, index, feature_weights, mu, window)

    def unigram_matches(self, query_terms: List[str], doc_id: str) -> float:
        """Returns unigram matches based on smoothed entity language model.

        Args:
            query_terms: List of query terms
            doc_id: Document ID for the document we wish to score

        Returns:
            Score for unigram matches for document with doc ID.
        """
        score = 0
        doc_terms = self.collection.get(doc_id)
        cqe = 0
        qfreq = 0
        tot_terms = 0
        for qterm in query_terms:
            for doc_ent in self.collection:
                d = self.collection.get(doc_ent)
                for terms in d:
                    print(qterm, terms)
                    if (qterm == terms) and (doc_ent == doc_id):
                        cqe += 1
                    if qterm == terms:
                        qfreq += 1
                    tot_terms += 1
            print(tot_terms)
            prelog = ((cqe + self.mu*(qfreq/tot_terms)) / (len(doc_terms)+ self.mu))
            score += math.log(prelog) if prelog != 0 else 0
            cqe = 0
            qfreq = 0
            tot_terms = 0          
                        
        return score

    def ordered_bigram_matches(self, query_terms: List[str], doc_id):
        """Returns ordered bigram matches based on smoothed entity language
        model.

        Args:
            query_terms: List of query terms
            doc_id: Document ID for the document we wish to score

        Returns:
            Score for ordered bigram matches for document with doc ID.
        """
        bigrams = []
        for ind in range(0,len(query_terms)-1):
            if (ind+1) > len(query_terms):
                break
            bigrams.append([query_terms[ind],query_terms[ind+1]])
        score = 0
        bfreq = 0
        bce = 0
        tot_terms = 0
        dd = 0
        for doc_ent in self.collection:
            d = self.collection.get(doc_ent)
            tot_terms += len(d)
        print("total length of docs: ", dd)
        for bigram in bigrams:
            print(bigram)
            for doc_ent in self.collection:
                d = self.collection.get(doc_ent)
                for tind  in range(0, len(d)-1):
                    if tind+1 > len(d):
                        break
                    if d[tind] == bigram[0] and d[tind+1] == bigram[1]:
                        bfreq += 1
                    if d[tind] == bigram[0] and d[tind+1] == bigram[1] and doc_ent == doc_id:
                        bce += 1
            print("bce", bce)
            print("bfreq", bfreq)
            print("tot_terms", tot_terms)
            prelog = (bce + self.mu*(bfreq/tot_terms)) / (len(self.collection.get(doc_id)) + self.mu)
            score += math.log(prelog) if prelog != 0 else 0
            bfreq = 0
            bce = 0
        return score

    def unordered_bigram_matches(self, query_terms: List[str], doc_id):
        """Returns unordered bigram matches based on smoothed entity language
        model.

        Args:
            query_terms: List of query terms
            doc_id: Document ID for the document we wish to score

        Returns:
            Score for unordered bigram matches for document with doc ID.
        """
        f_O = 0
        collection = self.collection[doc_id]
        print(query_terms)
        for i in range(0, len(query_terms)-1):
            C_w = 0
            le = len(collection)
            for j in range(0, le-self.window+1):
                window = collection[j:j+self.window]
                if query_terms[i] in window and query_terms[i+1] in window:
                    C_w += 1
            n_query_collection = 0
            N = 0
            for terms in self.collection.values():
                N += len(terms)
                for j in range(0, len(terms)-self.window+1):
                    window = terms[j:j+self.window]
                    if query_terms[i] == query_terms[i+1]:
                        if window.count(query_terms[i]) > 1:
                            n_query_collection += 1
                    elif query_terms[i] in window and query_terms[i+1] in window:
                        n_query_collection += 1
            P_w = n_query_collection/N
            result = (C_w+self.mu*P_w)/(le+self.mu)
            f_O += math.log(result) if result != 0 else 0
        return f_O


class FSDMScorer(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        feature_weights=[0.85, 0.1, 0.05],
        mu: float = 100,
        window: int = 3,
        fields: List[str] = ["title", "body", "anchors"],
        field_weights: List[float] = [0.2, 0.7, 0.1],
    ):
        """SDM scorer. This scorer can be applied on a single field only.

        Args:
            collection: Collection of documents. Needed to calculate document
                statistical information.
            index: Index to use for calculating scores.
            field (optional): Single field to use in scoring.. Defaults to None.
            fields (optional): List of fields to use in scoring. Defaults to
                None.
        """
        super().__init__(collection, index, feature_weights, mu, window)
        self.fields = fields
        self.field_weights = field_weights

    def unigram_matches(self, query_terms: List[str], doc_id: str) -> float:
        """Returns unigram matches based on smoothed entity language model.

        Args:
            query_terms: List of query terms
            doc_id: Document ID for the document we wish to score

        Returns:
            Score for unigram matches for document with doc ID.
        """
        # TODO
        return 0

    def ordered_bigram_matches(self, query_terms: List[str], doc_id):
        """Returns ordered bigram matches based on smoothed entity language
        model.

        Args:
            query_terms: List of query terms
            doc_id: Document ID for the document we wish to score

        Returns:
            Score for ordered bigram matches for document with doc ID.
        """

        # TODO
        return 0

    def unordered_bigram_matches(self, query_terms: List[str], doc_id):
        """Returns unordered bigram matches based on smoothed entity language
        model.

        Args:
            query_terms: List of query terms
            doc_id: Document ID for the document we wish to score

        Returns:
            Score for unordered bigram matches for document with doc ID.
        """
        # TODO
        return 0
collection_x = DocumentCollection(
    {
        "d1": {"body": ["t3", "t3", "t3", "t6", "t6"]},
        "d2": {"body": ["t1", "t2", "t3", "t3", "t6"]},
        "d3": {"body": ["t3", "t3", "t4", "t5"]},
        "d4": {"body": ["t4", "t5", "t6", "t6"]},
        "d5": {"body": ["t1", "t2", "t3", "t5"]},
    }
    )
index_1 = {
    "body": {
        "t1": [("d2", 1), ("d5", 1)],
        "t2": [("d2", 1), ("d5", 1)],
        "t3": [("d1", 3), ("d2", 2), ("d3", 2), ("d5", 1)],
        "t4": [("d3", 1), ("d4", 1)],
        "t5": [("d3", 1), ("d4", 1), ("d5", 1)],
        "t6": [("d1", 2), ("d2", 1), ("d4", 2)],
    }
}

if __name__ == "__main__":
    sc = SDMScorer(collection_x.get_field_documents("body"), index_1["body"])
    print(sc.unordered_bigram_matches(["t7", "t3", "t3"], "d1"))