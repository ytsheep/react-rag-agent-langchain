"""
RAG services.

A: baseline vector retrieval only.
C: hybrid retrieval over the existing Chroma corpus with parallel vector recall,
BM25 recall, fusion rerank, and top-3 context to the LLM.
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from rag.hybrid_retriever import HybridRetriever
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


def build_context(context_docs: list[Document]) -> str:
    chunks = []
    for index, doc in enumerate(context_docs, start=1):
        chunks.append(f"[REF {index}] content: {doc.page_content} | metadata: {doc.metadata}")
    return "\n".join(chunks)


class BaseRagSummarizeService:
    """Scheme A: original vector-only baseline."""

    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        return self.prompt_template | print_prompt | self.model | StrOutputParser()

    def retriever_docs(self, query: str, top_k: int | None = None) -> list[Document]:
        if top_k is None:
            return self.retriever.invoke(query)
        return self.vector_store.vector_store.similarity_search(query, k=top_k)

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)
        context = build_context(context_docs)
        return self.chain.invoke({"input": query, "context": context})


class HybridRagSummarizeService:
    """Scheme C: parallel recall + fusion rerank + top3."""

    def __init__(self):
        self.hybrid_retriever = HybridRetriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        return self.prompt_template | print_prompt | self.model | StrOutputParser()

    def retriever_docs(self, query: str, top_k: int | None = None) -> list[Document]:
        return self.hybrid_retriever.retrieve(query, final_k=top_k)

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)
        context = build_context(context_docs)
        return self.chain.invoke({"input": query, "context": context})


class RagSummarizeService(HybridRagSummarizeService):
    """Default app service: scheme C."""


if __name__ == "__main__":
    rag = RagSummarizeService()
    print(rag.rag_summarize("小户型适合哪些扫地机器人？"))
