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
            "question-answering", model=path, tokenizer=path, **kwargs
        )
