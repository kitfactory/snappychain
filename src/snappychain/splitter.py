from langchain_core.runnables import RunnableLambda
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter
)
from langchain.schema import Document
from typing import List, Optional


def markdown_text_splitter(text_key: str = "text") -> RunnableLambda:
    """
    Split each Document's text using MarkdownTextSplitter.
    data["_session"]["documents"] にある各 Document の page_content を MarkdownTextSplitter を使用して分割します。

    Returns:
        RunnableLambda: A lambda that updates data["_session"]["documents"] with new Document objects
                        created from the split chunks.
    """
    def inner(data):
        docs = data.get("_session", {}).get("documents", [])
        new_docs = []
        splitter = MarkdownTextSplitter()
        for doc in docs:
            chunks = splitter.split_text(doc.page_content)
            for chunk in chunks:
                new_doc = Document(page_content=chunk, metadata=doc.metadata)
                new_docs.append(new_doc)
        if "_session" not in data or not isinstance(data["_session"], dict):
            data["_session"] = {}
        data["_session"]["documents"] = new_docs
        return data
    return RunnableLambda(inner)


def json_text_splitter(text_key: str = "text") -> RunnableLambda:
    """
    Split each Document's text using RecursiveCharacterTextSplitter with text settings.
    data["_session"]["documents"] にある各 Document の page_content を テキスト用の設定を持つ RecursiveCharacterTextSplitter を使用して分割します。

    Returns:
        RunnableLambda: A lambda that updates data["_session"]["documents"] with new Document objects
                        created from the split chunks.
    """
    def inner(data):
        docs = data.get("_session", {}).get("documents", [])
        new_docs = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["}}", "}", "]", "\n", " ", ""]
        )
        for doc in docs:
            chunks = splitter.split_text(doc.page_content)
            for chunk in chunks:
                new_doc = Document(page_content=chunk, metadata=doc.metadata)
                new_docs.append(new_doc)
        if "_session" not in data or not isinstance(data["_session"], dict):
            data["_session"] = {}
        data["_session"]["documents"] = new_docs
        return data
    return RunnableLambda(inner)


def python_text_splitter() -> RunnableLambda:
    """
    Split each Document's text using PythonCodeTextSplitter.
    data["_session"]["documents"] にある各 Document の page_content を PythonCodeTextSplitter を使用して分割します。

    Returns:
        RunnableLambda: A lambda that updates data["_session"]["documents"] with new Document objects
                        created from the split chunks.
    """
    def inner(data):
        docs = data.get("_session", {}).get("documents", [])
        new_docs = []
        splitter = PythonCodeTextSplitter()
        for doc in docs:
            chunks = splitter.split_text(doc.page_content)
            for chunk in chunks:
                new_doc = Document(page_content=chunk, metadata=doc.metadata)
                new_docs.append(new_doc)
        if "_session" not in data or not isinstance(data["_session"], dict):
            data["_session"] = {}
        data["_session"]["documents"] = new_docs
        return data
    return RunnableLambda(inner)


def split_text(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separator: str = "\n\n"
) -> RunnableLambda:
    """
    Split each Document's text using CharacterTextSplitter.
    各Documentのテキストを CharacterTextSplitter を使用して分割します。

    Args:
        chunk_size (int): Maximum size of chunks to return
                         返すチャンクの最大サイズ
        chunk_overlap (int): Overlap in characters between chunks
                           チャンク間の重複文字数
        separator (str): Separator to use between chunks
                        チャンク間の区切り文字

    Returns:
        RunnableLambda: A lambda that splits documents and updates data["_session"]["documents"]
                        ドキュメントを分割し、data["_session"]["documents"]を更新するラムダ
    """
    def inner(data):
        docs = data.get("_session", {}).get("documents", [])
        if not docs:
            return data

        text_splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        split_docs = []
        for doc in docs:
            split_docs.extend(text_splitter.split_documents([doc]))

        data["_session"]["documents"] = split_docs
        return data
    return RunnableLambda(inner)


def recursive_split_text(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None
) -> RunnableLambda:
    """
    Split each Document's text using RecursiveCharacterTextSplitter.
    各Documentのテキストを RecursiveCharacterTextSplitter を使用して分割します。

    Args:
        chunk_size (int): Maximum size of chunks to return
                         返すチャンクの最大サイズ
        chunk_overlap (int): Overlap in characters between chunks
                           チャンク間の重複文字数
        separators (List[str]): List of separators to use for splitting, in order of priority
                               分割に使用する区切り文字のリスト（優先順位順）

    Returns:
        RunnableLambda: A lambda that splits documents and updates data["_session"]["documents"]
                        ドキュメントを分割し、data["_session"]["documents"]を更新するラムダ
    """
    def inner(data):
        docs = data.get("_session", {}).get("documents", [])
        if not docs:
            return data

        # Default separators if none provided
        # デフォルトの区切り文字（指定がない場合）
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]

        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        split_docs = []
        for doc in docs:
            split_docs.extend(text_splitter.split_documents([doc]))

        data["_session"]["documents"] = split_docs
        return data
    return RunnableLambda(inner)