import sys
import os
import pandas as pd
import numpy as np

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

from . import BaseQA
from ..retrievers import BaseRetriever


class Llama3(BaseQA):

    def __init__(
        self,
        docs_df: pd.DataFrame,
        retriever: BaseRetriever,
        model_path: str,
        prompt_template: PromptTemplate = None,
        **kwargs
    ):
        """
        Initializes the class instance

        Arguments: docs_df : pd.DataFrame
                        Documents used for the open-book information retrieval. The dataframe
                        must have the following fields: "url", "title" and "content".

                    model_path: str
                        The path to the pretrained model used for the question answering
                        task

                    prompt_template (optional): langchain_core.prompts.PromptTemplate
                        Prompt template. Must have placeholders for the context and the question.

        Returns:
            None
        """
        if prompt_template is None:
            prompt_template = PromptTemplate.from_template(
                "Using the following context: {context},\n answer: {question}"
            )
        self._prompt_template = prompt_template
        super().__init__(docs_df, retriever, model_path, **kwargs)

    def _model_init(self, path, **kwargs):
        """
        Initializes the model

        Arguments: path: str
                       Path to the model
                   **kwargs

        Returns:
            None
        """
        with suppress_stdout_stderr():
            self._model = LlamaCpp(model_path=path, **kwargs)

    def set_prompt_template(self, pt: PromptTemplate):
        """
        Set a new prompt template

        Arguments: pt : langchain_core.prompts.PromptTemplate

        Returns:
            None
        """
        self._prompt_template = pt

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

    def answer(self, query: str, k_docs: int = 3) -> dict:
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
        ans = self._model.invoke(
            self._prompt_template.format(
                question=query, context=context_dict["context"]
            )
        )
        ans_dict = dict(answer=ans, **context_dict)
        return ans_dict


class suppress_stdout_stderr(object):
    # extracted from https://github.com/abetlen/llama-cpp-python/issues/478
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()
