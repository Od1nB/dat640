import abc
from collections import UserDict as DictClass
from typing import Dict, List

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
