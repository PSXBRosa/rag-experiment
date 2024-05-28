from __future__ import annotations
import re
from langchain_core.prompts import PromptTemplate
from .models import BaseQA


class BaseWrapper:
    def __init__(self, qa_model: BaseQA | BaseWrapper):
        assert not isinstance(
            qa_model, AbsoluteAnswerWrapper
        ), "'AbsluteAnswerWrapper' is not a valid qa_model"

        self._qa_model = qa_model

    def answer(self, query: str) -> dict:
        raise NotImplementedError


class AbsoluteAnswerWrapper:
    def __init__(self, qa_model: BaseQA | BaseWrapper):
        self._qa_model = qa_model

    def answer(self, query: str) -> str:
        return self._qa_model.answer(query)["answer"]


class MinimumCertaintyWrapper(BaseWrapper):
    def __init__(
        self,
        qa_model: BaseQA | BaseWrapper,
        confidence_thr: int = 0.01,
        *args,
        **kwargs,
    ):
        super().__init__(qa_model, *args, **kwargs)
        self._thr = confidence_thr

    def answer(self, query: str) -> dict:
        ans_dict = self._qa_model.answer(query)

        if ans_dict["score"] < self._thr:
            ans_dict[
                "answer"
            ] = "Je suis désolée, je ne sais pas comment répondre à cette question."
        return ans_dict


class AppendURLWrapper(BaseWrapper):
    def answer(self, query: str) -> dict:
        ans_dict = self._qa_model.answer(query)
        start, end, ranges, urls = (
            ans_dict["start"],
            ans_dict["end"],
            ans_dict["ranges"],
            ans_dict["urls"],
        )

        for idx, (range_start, range_end) in enumerate(ranges):
            if start >= range_start and end <= range_end:
                break

        src_doc_url = urls[idx]
        ans_dict["answer"] += "\n\n Pour plus d'informations, accédez " + src_doc_url
        return ans_dict


class FetchEntireSentenceWrapper(BaseWrapper):
    def answer(self, query: str) -> dict:
        ans_dict = self._qa_model.answer(query)
        start, end, context = ans_dict["start"], ans_dict["end"], ans_dict["context"]

        def find_newline(rng):
            ans = None
            for idx in rng:
                if context[idx] != "\n":
                    continue
                ans = idx
                break
            return ans

        start_of_paragraph = find_newline(range(start, -1, -1))
        start_of_paragraph = start if start_of_paragraph is None else start_of_paragraph

        end_of_paragraph = find_newline(range(end, len(context)))
        end_of_paragraph = end if end_of_paragraph is None else end_of_paragraph

        paragraph = context[start_of_paragraph + 1 : end_of_paragraph]
        sentences = re.split(r"(?<=[.!?])\s", paragraph)

        for sentence in sentences:
            if ans_dict["answer"] in sentence:
                ans_dict["answer"] = sentence
        return ans_dict


class SetPromptTemplate(BaseWrapper):
    def __init__(self, qa_model: BaseQA | BaseWrapper, prompt_text: str):
        super().__init__(qa_model)
        pt = PromptTemplate.from_template(prompt_text)
        self._qa_model.set_prompt_template(pt)

    def answer(self, *args, **kwargs):
        return self._qa_model.answer(*args, **kwargs)
