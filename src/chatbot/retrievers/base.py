import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline


class BaseRetriever:

    def __init__(self, docs_df, **kwargs):
        """
        Initialize the retriever object

        Argumets: docs_df: pd.DataFrame
                      Documents used for the retrieval
                  **kwargs

        Returns:
            None
        """
        self._df = docs_df

    def get_context(self, query: str, k_docs: int) -> dict:
        """
        retrieves the documents

        Parameters: query : str
                        input query

                    k_docs : int
                        number of best ranking documents to be fed forward into the text
                        comprehension model

        Returns: ans : dict
                    A dictionary containing the necessary keys
        """
        raise NotImplementedError
