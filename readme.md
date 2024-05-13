# Assistance Free ChatBot

## Installation

```pip install -r requirements.txt```

## How to use the chatbot

from the root folder:

```python src/main.py configs/ifidf-camembert.yaml```

## Theorical explaination

Given the data presented and the project goals, It was identified that the most suitable system would be similar to an Open-Domain Question Answering System. These systems are a type of artifitial intelligence system that can answer questions in natural language The “open-domain” part refers to the lack of the relevant context for any asked question. The model only takes as the input the question but no context is provided, where the answer for the question is likely to be found. In the case when both the question and the context are provided, the task is known as Reading comprehension (RC).

These systems may work with or without access to an external collection of documents from where to obtain the answer, and thus are referred to as open-book or closed-book question answering, respectively.

In the context of the present problem, we could consider the three following frameworks (order in increasing complexity):

1. Retriever-Reader     (open-book)
2. Retriever-Generator  (open-book)
3. Generator            (closed-book)

The first two frameworks require an external source of knowledge and are thus classified as "open-book" models. These two make use of a retriever element, which is responsible for retrieving the context from the context-base and feeding it forward into the system, which will be responsible for extracting the answer and returning it in natural language.

The third framework, the "Generator", represents the set of auto-regressive models. These models (like T5, GPT etc) are capable of storing information during training and later including that information in the generated answers on evaluation and because of that are often the preferred framework.

Although the third framework is often the one chosen due to its recent success cases (GP3, Bard, LLAMA etc), it has some big drawbacks, namely:

* model size often too big for most consumer machines,
* or the need of an existing pre-trained model suitable for the task;
* longer time to train;
* harder to implement.

Because of those points and the limited resources and time, the third and second frameworks were discarted.
---

### Retriever

For the retrieveing problem, a tf-idf encoder was used, using the scikit-learn module. The retriever task is to obtain the document containing the answer for any given prompt from the collection of documents (in this case, the sqlite3 database).

### Reader

For the reader, a [https://huggingface.co/etalab-ia/camembert-base-squadFR-fquad-piaf](pretrained) version of the [https://camembert-model.fr/](camembert) model was used for reading the context and retrieving the answer.

## References

1. [https://lilianweng.github.io/posts/2020-10-29-odqa/#openai-api-example](https://lilianweng.github.io/posts/2020-10-29-odqa/#openai-api-example)

2. [https://arxiv.org/abs/1704.00051](https://arxiv.org/abs/1704.00051)

3. [https://huggingface.co/etalab-ia/
camembert-base-squadFR-fquad-piaf](https://huggingface.co/etalab-ia/camembert-base-squadFR-fquad-piaf)

4. [https://camembert-model.fr/](https://camembert-model.fr/)
