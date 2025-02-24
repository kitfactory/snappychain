from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
from typing import List, Optional
import os

def openai_embedding() -> RunnableLambda:
    """
    Create and store OpenAI embeddings model in session data.
    OpenAIのembeddingsモデルを作成し、セッションデータに保存します。

    Returns:
        RunnableLambda: A lambda that stores OpenAIEmbeddings model in data["_session"]["model"].
                        OpenAIEmbeddingsモデルをdata["_session"]["model"]に保存するラムダ。
    """
    def inner(data):
        if "_session" not in data:
            data["_session"] = {}
        data["_session"]["model"] = OpenAIEmbeddings()
        return data
    return RunnableLambda(inner)


def ollama_embedding(model_name: str = "llama2") -> RunnableLambda:
    """
    Create and store Ollama embeddings model in session data.
    Ollamaのembeddingsモデルを作成し、セッションデータに保存します。

    Args:
        model_name (str): Name of the Ollama model to use.
                         使用するOllamaモデルの名前。

    Returns:
        RunnableLambda: A lambda that stores OllamaEmbeddings model in data["_session"]["model"].
                        OllamaEmbeddingsモデルをdata["_session"]["model"]に保存するラムダ。
    """
    def inner(data):
        if "_session" not in data:
            data["_session"] = {}
        data["_session"]["model"] = OllamaEmbeddings(model=model_name)
        return data
    return RunnableLambda(inner)


def add_documents_to_vector_store() -> RunnableLambda:
    """
    Add documents to the vector store from session data.
    セッションデータからドキュメントをベクトルストアに追加します。

    Returns:
        RunnableLambda: A lambda that adds documents to the vector store
                        ドキュメントをベクトルストアに追加するラムダ
    """
    def inner(data):
        docs = data.get("_session", {}).get("documents", [])
        if not docs:
            return data

        vector_store = data.get("_session", {}).get("vector_store")
        if not vector_store:
            raise ValueError("No vector store found in session data.")

        vector_store.add_documents(docs)
        return data
    return RunnableLambda(inner)


def persist_vector_store(persist_dir: str) -> RunnableLambda:
    """
    Persist vector store from session data to disk.
    セッションデータのベクトルストアをディスクに永続化します。

    Args:
        persist_dir (str): Directory to persist the vector store
                          ベクトルストアを永続化するディレクトリ

    Returns:
        RunnableLambda: A lambda that persists the vector store
                        ベクトルストアを永続化するラムダ
    """
    def inner(data):
        vector_store = data.get("_session", {}).get("vector_store")
        if not vector_store:
            raise ValueError("No vector store found in session data.")

        os.makedirs(persist_dir, exist_ok=True)
        vector_store.save_local(persist_dir)
        return data
    return RunnableLambda(inner)


def query_vector_store(query: str, k: int = 4) -> RunnableLambda:
    """
    Query the vector store from session data for similar documents.
    セッションデータのベクトルストアに類似ドキュメントを問い合わせます。

    Args:
        query (str): Query text
                    問い合わせテキスト
        k (int): Number of documents to return
                返すドキュメントの数

    Returns:
        RunnableLambda: A lambda that queries the vector store and returns similar documents
                        ベクトルストアを検索し類似ドキュメントを返すラムダ
    """
    def inner(data):
        vector_store = data.get("_session", {}).get("vector_store")
        if not vector_store:
            raise ValueError("No vector store found in session data.")

        similar_docs = vector_store.similarity_search(query, k=k)
        if "_session" not in data:
            data["_session"] = {}
        data["_session"]["similar_documents"] = similar_docs
        return data
    return RunnableLambda(inner)
