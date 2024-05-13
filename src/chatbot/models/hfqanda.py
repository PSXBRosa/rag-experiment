import pandas as pd
import numpy as np

from transformers import pipeline

from . import BaseQA
from ..retrievers import BaseRetriever


class HFQandA(BaseQA):

    def _model_init(self, path, **kwargs):
        """
        Initialize the model

        Arguments: path: str
                       Path to the model
                   **kwargs

        Returns:
            None
        """
        self._model = pipeline(
            'question-answering',
            model=path,
            tokenizer=path,
            **kwargs
        )

    def get_context(self, query: str, k_docs: int) -> dict:
        """
        Information retrieval part of the model.

        Parameters: query : str
                        input query

                    k_docs : int
                        number of best ranking documents to be fed forward into the text
                        comprehension model

        Returns: ans : dict
                    A dictionary containing the following keys: "context", "ranges" and "urls".
                    The context is the concatenation of the top k documentss while "ranges"
                    and "urls" are respectively the start and end indexes of each document in-
                    side the context field and the urls related to those documents.
        """

        # add the urls to the answer
        ans = super().get_context(query, k_docs)
        doc_urls = [self._df["url"][i] for i in ans["ids"]]
        return dict(urls=doc_urls, **ans)
