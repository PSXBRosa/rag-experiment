import pandas as pd
import numpy as np
from .. import retrievers


class BaseQA:

    def __init__(
            self,
            docs_df: pd.DataFrame,
            retriever: retrievers.BaseRetriever,
            model_path: str,
            **kwargs
        ):
        """
        Initializes the class parameters

        Arguments: docs_df: pd.DataFrame
                       Documents used for the information retrieval
                  
                   retriever: chatbot.retrievers.BaseRetriever
                       Retriever object with an implemented get_context method

                   model_path: str
                       The path to the pretrained model used for the task

        Returns:
            None
        """
        self._df = docs_df
        self._retriever = retriever
        self._model_init(model_path, **kwargs.get("model_init_kwargs", {}))

    def answer(self, query: str, k_docs: int=3) -> dict:
        """
        Executes the information retrieval and text comprehension

        Arguments:  query : str
                        Input query

                    k_docs : int
                        Number of best ranking documents to be fed forward into the text
                        comprehension model

        Returns: ans : dict
                    A dictionary containing the model output plus the information retrieval fields
        """
        context_dict = self.get_context(query, k_docs)

        inp_dict = {
            "question": query,
            "context": context_dict["context"]
        }

        ans_dict = self._model(inp_dict)
        ans_dict = dict(ans_dict, **context_dict)
        return ans_dict

    def get_context(self, query: str, k_docs: int) -> dict:
        """
        Information retrieval part of the model.

        Parameters: query : str
                        input query

                    k_docs : int
                        number of best ranking documents to be fed forward into the text
                        comprehension model

        Returns: ans : dict
                    A dictionary with the necessary output
        """
        return self._retriever.get_context(query, k_docs)

    def _model_init(self, path, **kwargs):
        """
        Initialize the model

        Arguments: path: str
                       Path to the model
                   **kwargs

        Returns:
            None
        """
        raise NotImplementedError

