import pandas as pd
import numpy as np

from collections.abc import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer

from . import BaseRetriever

class TfIdf(BaseRetriever):

    def __init__(self, docs_df: pd.DataFrame, indexing: Iterable, **kwargs):
        """
        Initialize the retriever object

        Argumets: docs_df: pd.DataFrame
                      Documents used for the retrieval
                  indexing: Iterable
                      Indexing for the vectorizer
                  **kwargs

        Returns:
            None
        """
        super().__init__(docs_df)
        self._tfidf_vectorizer = TfidfVectorizer(**kwargs)
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(indexing)

    def get_context(self, query: str, k_docs: int) -> dict:
        """
        Retrieves the k first relevant documents

        Parameters: query : str
                        input query

                    k_docs : int
                        number of best ranking documents to be fed forward into the text
                        comprehension model

        Returns: ans : dict
                    A dictionary containing the necessary keys
        """
        query_vector = self._tfidf_vectorizer.transform([query])
        query_vector = query_vector.toarray()[0]

        document_scores = self._tfidf_matrix.dot(query_vector)
        sorted_indices = document_scores.argsort()[::-1][:k_docs]

        doc_contents = [self._df["content"][i] for i in sorted_indices]
        context = "\n".join(doc_contents)

        docs_len = [len(self._df["content"][i]) for i in sorted_indices]

        # [idx doc start, idx doc ends[
        docs_range = [(i + k, j + k) for i, j, k in zip([0] + docs_len, docs_len, range(len(docs_len)))]

        return {"ids": sorted_indices, "context": context, "ranges": docs_range}

