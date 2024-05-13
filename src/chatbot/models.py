import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

class QuestionAnswering:

    def __init__(
            self,
            df: pd.DataFrame,
            pretrained: str='etalab-ia/camembert-base-squadFR-fquad-piaf',
            **kwargs
        ):
        """
        Initializes the class instance

        Parameters: df : pd.DataFrame
                        Documents used for the open-book information retrieval. The dataframe
                        must have the following fields: "url", "title" and "content".

                    pretrained : str
                        The reference to the pretrained model used for the question answering
                        task

        Returns: None
        """

        pipeline_kwargs = kwargs.get("pipeline_kwargs", {})
        vectorizer_kwargs = kwargs.get("vectorizer_kwargs", {})

        self._df = df

        self._nlp = pipeline(
            'question-answering',
            model=pretrained,
            tokenizer=pretrained,
            **pipeline_kwargs
        )

        if "ngram_range" not in kwargs:
            vectorizer_kwargs["ngram_range"] = (1,2)
        if "strip_accents" not in kwargs:
            vectorizer_kwargs["strip_accents"] = "unicode"
        self._tfidf_vectorizer = TfidfVectorizer(**vectorizer_kwargs)

        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(df["title"])


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

        query_vector = self._tfidf_vectorizer.transform([query])
        query_tfidf_scores = query_vector.toarray()[0]

        document_scores = self._tfidf_matrix.dot(query_tfidf_scores)
        sorted_indices = document_scores.argsort()[::-1][:k_docs]

        doc_contents = [self._df["content"][i] for i in sorted_indices]
        context = "\n".join(doc_contents)

        lens = [len(self._df["content"][i]) for i in sorted_indices]
        lens = np.cumsum([0] + lens)

        doc_ranges = [(i + k, j + k) for i, j, k in zip(lens, lens[1:], range(len(lens) - 1))]
        doc_urls = [self._df["url"][i] for i in sorted_indices]

        return {"context": context, "ranges": doc_ranges, "urls": doc_urls}


    def answer(self, query: str, k_docs: int=3) -> dict:
        """
        Executes the information retrieval and text comprehension

        Parameters: query : str
                        input query

                    k_docs : int
                        number of best ranking documents to be fed forward into the text
                        comprehension model

        Returns: ans : dict
                    A dictionary containing the model output plus the information retrieval fields
        """
        context_dict = self.get_context(query, k_docs)

        inp_dict = {
            "question": query,
            "context": context_dict["context"]
        }

        ans_dict = self._nlp(inp_dict)
        ans_dict = dict(ans_dict, **context_dict)
        return ans_dict
