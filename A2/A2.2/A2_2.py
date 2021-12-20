import abc
import math
from collections import Counter, defaultdict
from typing import Dict, List
from collections import UserDict as DictClass

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
        field: str = None,
        fields: List[str] = None,
    ):
        """Interface for the scorer class.

        Args:
            collection: Collection of documents. Needed to calculate document
                statistical information.
            index: Index to use for calculating scores.
            field (optional): Single field to use in scoring.. Defaults to None.
            fields (optional): List of fields to use in scoring. Defaults to
                None.

        Raises:
            ValueError: Either field or fields need to be specified.
        """
        self.collection = collection
        self.index = index

        if not (field or fields):
            raise ValueError("Either field or fields have to be defined.")

        self.field = field
        self.fields = fields

        # Score accumulator for the query that is currently being scored.
        self.scores = None

    def score_collection(self, query_terms: List[str]):
        """Scores all documents in the collection using term-at-a-time query
        processing.

        Params:
            query_term: Sequence (list) of query terms.

        Returns:
            Dict with doc_ids as keys and retrieval scores as values.
            (It may be assumed that documents that are not present in this dict
            have a retrival score of 0.)
        """
        self.scores = defaultdict(float)  # Reset scores.
        query_term_freqs = Counter(query_terms)

        for term, query_freq in query_term_freqs.items():
            self.score_term(term, query_freq)

        return self.scores

    @abc.abstractmethod
    def score_term(self, term: str, query_freq: int):
        """Scores one query term and updates the accumulated document retrieval
        scores (`self.scores`).

        Params:
            term: Query term
            query_freq: Frequency (count) of the term in the query.
        """
        raise NotImplementedError


class SimpleScorer(Scorer):
    def score_term(self, term: str, query_freq: int) -> None:
        freqlist = self.index.get(self.field).get(term)
        if freqlist:
            for docs in freqlist:
                self.scores[docs[0]] = int(docs[1]*query_freq)


class ScorerBM25(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: str = "body",
        b: float = 0.75,
        k1: float = 1.2,
    ) -> None:
        super(ScorerBM25, self).__init__(collection, index, field)
        self.b = b
        self.k1 = k1

    def score_term(self, term: str, query_freq: int) -> None:
        freqlist = self.index.get(self.field).get(term)
        totlen = 0
        for _, documents in self.collection.items():

            totlen += len(documents.get(self.field))
        avgdoclen = totlen/len(self.collection)
        print(avgdoclen)
        if freqlist:
            for docs in freqlist:
                self.scores[docs[0]] += \
                    (docs[1]*(1+self.k1))/\
                    (docs[1]+self.k1*(1 - self.b + self.b *((len(self.collection.get(docs[0]).get(self.field)))/(avgdoclen))))\
                    * math.log(len(self.collection.keys())/len(self.index.get(self.field).get(term))) #idft ok i think

class ScorerLM(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: str = "body",
        smoothing_param: float = 0.1,
    ):
        super(ScorerLM, self).__init__(collection, index, field)
        self.smoothing_param = smoothing_param

    def score_term(self, term: str, query_freq: int) -> None:
        termocc = 0
        totdoclen = 0
        lamb = self.smoothing_param
        for dockey, value in self.collection.items():
            if self.index.get(self.field).get(term):
                for freqs in self.index.get(self.field).get(term): 
                    if freqs[0] == dockey:
                        termocc += freqs[1]
            totdoclen += len(value.get(self.field))
        relativeP = termocc/totdoclen
        for dockey, doc in self.collection.items():
            doclen = len(doc.get(self.field))
            doc_term_freq = 0
            for docs in self.index.get(self.field).get(term):
                if docs[0] == dockey:
                    doc_term_freq = docs[1]
            self.scores[dockey] += query_freq * math.log((1-lamb)*(doc_term_freq/doclen)+(lamb*relativeP))


class ScorerBM25F(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        fields: List[str] = ["title", "body"],
        field_weights: List[float] = [0.2, 0.8],
        bi: List[float] = [0.75, 0.75],
        k1: float = 1.2,
    ) -> None:
        super(ScorerBM25F, self).__init__(collection, index, fields=fields)
        self.field_weights = field_weights
        self.bi = bi
        self.k1 = k1

    def score_term(self, term: str, query_freq: int) -> None:
        totdocs = len(self.collection.keys())
        n = len(self.index.get("body").get(term))
        idtf = math.log(totdocs/n)
        for dockey, _ in self.collection.items():
            ctda = 0
            for fieldi in range(0,len(self.fields)):
                field, field_weight = self.fields[fieldi], self.field_weights[fieldi]
                ctd = 0
                totlenfield = 0
                for docs in self.collection.values():
                    totlenfield+= len(docs.get(field))
                avgdli = totlenfield/totdocs
                if self.index.get(field).get(term):
                    for freq in self.index.get(field).get(term):
                        if freq[0] == dockey:
                            ctd += freq[1]
                di = len(self.collection.get(dockey).get(field))
                Bi = (1- self.bi[fieldi]+self.bi[fieldi]*(di/avgdli))
                ctda += field_weight* (ctd/Bi)
            self.scores[dockey] += (ctda/(self.k1+ctda))* idtf


class ScorerMLM(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        fields: List[str] = ["title", "body"],
        field_weights: List[float] = [0.2, 0.8],
        smoothing_param: float = 0.1,
    ):
        super(ScorerMLM, self).__init__(collection, index, fields=fields)
        self.field_weights = field_weights
        self.smoothing_param = smoothing_param

    def score_term(self, term: str, query_freq: float) -> None:
        lamb = self.smoothing_param
        totfreq = [0]*len(self.fields)
        tot = [0]*len(self.fields)
        for dockey, docfielddic in self.collection.items():
            for fieldi in range(0, len(self.fields)):
                field = self.fields[fieldi]
                tot[fieldi] += len(docfielddic.get(field))
                if self.index.get(field).get(term):
                    for freq in self.index.get(field).get(term):
                        if freq[0] == dockey:
                            totfreq[fieldi] += freq[1]
        for dockey, docfielddic in self.collection.items():
            Ptheta = 0
            for fieldi in range(0,len(self.fields)):
                field, field_weight = self.fields[fieldi], self.field_weights[fieldi]
                emptfreqtot = len(self.collection.get(dockey).get(field))
                empttot = 0
                if self.index.get(field).get(term):
                    for freq in self.index.get(field).get(term):
                        if freq[0] == dockey:
                            empttot += freq[1]
                pt = empttot/emptfreqtot
                pc = totfreq[fieldi]/tot[fieldi]
                Ptheta += field_weight * ((1-lamb)*(pt)+(lamb*pc))
            self.scores[dockey] += query_freq * math.log(Ptheta)