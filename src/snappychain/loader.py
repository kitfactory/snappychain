from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader
)
from typing import Optional, List
import os
from onelogger import Logger

logger = Logger.get_logger(__name__)

def text_load(file_paths: List[str], encoding: str = "utf-8") -> RunnableLambda:
    """
    Returns a RunnableLambda that loads text documents from multiple files using LangChain's TextLoader.
    LangChainのTextLoaderを使用して、複数ファイルからテキストドキュメントを読み込むRunnableLambdaを返します.

    Args:
        file_paths (List[str]): A list of paths to text files.
                               テキストファイルのパスのリスト。
        encoding (str): The encoding to use when reading the files. Defaults to 'utf-8'.
                        ファイル読み込み時に使用するエンコーディング。デフォルトは 'utf-8'です。

    Returns:
        RunnableLambda: A RunnableLambda that loads documents and appends them to data[_session]["documents"].
                        ドキュメントを読み込み、data[_session]["documents"]に追加するRunnableLambdaを返します.
    """
    def inner(data):
        # Ensure that _session key exists in data // dataに_sessionキーが存在することを保証します
        if "_session" not in data or not isinstance(data["_session"], dict):
            data["_session"] = {}
        # Ensure that documents key exists in data[_session] // data[_session]にdocumentsキーが存在することを保証します
        if "documents" not in data["_session"] or not isinstance(data["_session"]["documents"], list):
            data["_session"]["documents"] = []
        for file_path in file_paths:
            loader = TextLoader(file_path, encoding=encoding)  # Initialize TextLoader with the file path and encoding // ファイルパスとエンコーディングでTextLoaderを初期化します
            documents = loader.load()       # Load the documents // ドキュメントを読み込みます
            data["_session"]["documents"].extend(documents)  # Append loaded documents to the session's documents list // 読み込んだドキュメントをセッションのリストに追加します
        return data
    return RunnableLambda(inner)

def pypdf_load(file_paths: List[str]) -> RunnableLambda:
    """
    Returns a RunnableLambda that loads PDF documents from multiple files using LangChain's PyPDFLoader.
    LangChainのPyPDFLoaderを使用して、複数ファイルからPDFドキュメントを読み込むRunnableLambdaを返します.

    Args:
        file_paths (List[str]): A list of paths to PDF files.
                               PDFファイルのパスのリスト。

    Returns:
        RunnableLambda: A RunnableLambda that loads documents and appends them to data[_session]["documents"].
                        ドキュメントを読み込み、data[_session]["documents"]に追加するRunnableLambdaを返します.
    """
    def inner(data):
        # Ensure that _session key exists in data // dataに_sessionキーが存在することを保証します
        if "_session" not in data or not isinstance(data["_session"], dict):
            data["_session"] = {}
        # Ensure that documents key exists in data[_session] // data[_session]にdocumentsキーが存在することを保証します
        if "documents" not in data["_session"] or not isinstance(data["_session"]["documents"], list):
            data["_session"]["documents"] = []
        for file_path in file_paths:
            loader = PyPDFLoader(file_path)  # Instantiate PyPDFLoader / PyPDFLoaderのインスタンスを生成
            documents = loader.load()      # Load the PDF document(s) / PDFドキュメントを読み込む
            data["_session"]["documents"].extend(documents)  # Append loaded documents to the session's documents list // 読み込んだドキュメントをセッションのリストに追加します
        return data
    return RunnableLambda(inner)

def markitdown_load(file_paths: List[str]) -> RunnableLambda:
    """
    Load files using MarkItDown and convert them into LangChain Document objects.
    MarkItDownを使用してファイルを変換し、LangChainのDocumentオブジェクトとして読み込みます。

    Args:
        file_paths (List[str]): List of file paths to convert. / 変換するファイルのパスのリスト。

    Returns:
        RunnableLambda: A lambda that loads documents and appends them to data['_session']["documents"].
                        ドキュメントを読み込み、data['_session']["documents"]に追加するRunnableLambdaを返します。
    """
    def inner(data):
        # Ensure that _session key exists in data
        if "_session" not in data or not isinstance(data["_session"], dict):
            data["_session"] = {}
        # Ensure that documents key exists in data[_session]
        if "documents" not in data["_session"] or not isinstance(data["_session"]["documents"], list):
            data["_session"]["documents"] = []
        
        from markitdown import MarkItDown
        from langchain.schema import Document
        
        md = MarkItDown(enable_plugins=False)  # Initialize MarkItDown / MarkItDownの初期化
        
        for file_path in file_paths:
            result = md.convert(file_path)  # Convert file using MarkItDown / MarkItDownを使用してファイルを変換
            # Create a Document with the converted text
            doc = Document(page_content=result.text_content, metadata={"source": file_path})
            data["_session"]["documents"].append(doc)  # Append the Document to the documents list
        return data
    return RunnableLambda(inner)

def get_chain_documents() -> RunnableLambda:
    """
    Load files using MarkItDown and convert them into LangChain Document objects.
    MarkItDownを使用してファイルを変換し、LangChainのDocumentオブジェクトとして読み込みます。

    Args:
        file_paths (List[str]): List of file paths to convert. / 変換するファイルのパスのリスト。

    Returns:
        RunnableLambda: A lambda that loads documents and appends them to data['_session']["documents"].
                        ラムダー関数を返します。
    """
    def inner(data):
        if "_session" not in data or not isinstance(data["_session"], dict):
            return None
        return data["_session"]["documents"]
    return RunnableLambda(inner)

def unstructured_markdown_load(file_paths: List[str]) -> RunnableLambda:
    """
    Load files using UnstructuredMarkdownLoader and convert them into LangChain Document objects.
    UnstructuredMarkdownLoaderを使用してファイルを変換し、LangChainのDocumentオブジェクトとして読み込みます。

    Args:
        file_paths (List[str]): List of file paths to convert. / 変換するファイルのパスのリスト。

    Returns:
        RunnableLambda: A lambda that loads documents and appends them to data['_session']["documents"].
                        ラムダー関数を返します。
    """
    def inner(data):
        if "_session" not in data or not isinstance(data["_session"], dict):
            data["_session"] = {}
        if "documents" not in data["_session"] or not isinstance(data["_session"]["documents"], list):
            data["_session"]["documents"] = []
        
        for file_path in file_paths:
            try:
                loader = UnstructuredMarkdownLoader(file_path)
                documents = loader.load()
                data["_session"]["documents"].extend(documents)
            except Exception as e:
                if data.get("_dev", False):
                    logger.error("\033[31mError loading markdown file %s: %s\033[0m", file_path, str(e))
                continue
        
        return data
    return RunnableLambda(inner)

def directory_load(
    directory_path: str,
    glob_pattern: str = "**/*.*",
    show_progress: bool = False,
    use_multithreading: bool = False
) -> RunnableLambda:
    """
    Load files from a directory using appropriate loaders based on file extensions.
    ディレクトリからファイルを読み込み、拡張子に応じて適切なローダーを使用してDocumentオブジェクトに変換します。

    Args:
        directory_path (str): Path to the directory containing files
                            ファイルを含むディレクトリのパス
        glob_pattern (str): Pattern to match files (default: "**/*.*" for all files recursively)
                          ファイルのマッチングパターン（デフォルト: "**/*.*" で再帰的に全ファイル）
        show_progress (bool): Whether to show a progress bar (default: False)
                            進捗バーを表示するかどうか（デフォルト: False）
        use_multithreading (bool): Whether to use multithreading for loading (default: False)
                                 マルチスレッドを使用するかどうか（デフォルト: False）

    Returns:
        RunnableLambda: A lambda that loads documents and appends them to data["_session"]["documents"]
                        ドキュメントを読み込んでdata["_session"]["documents"]に追加するラムダ
    """
    def get_loader_cls(file_path: str):
        """Get the appropriate loader class based on file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        loader_map = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.md': UnstructuredMarkdownLoader,
            # Add more mappings as needed
            # 必要に応じて他のマッピングを追加
        }
        return loader_map.get(ext)

    def inner(data):
        if "_session" not in data:
            data["_session"] = {}
        if "documents" not in data["_session"]:
            data["_session"]["documents"] = []

        try:
            # Create a loader for each supported extension
            # サポートされている拡張子ごとにローダーを作成
            for ext, loader_cls in {
                '.txt': TextLoader,
                '.pdf': PyPDFLoader,
                '.md': UnstructuredMarkdownLoader
            }.items():
                try:
                    loader = DirectoryLoader(
                        directory_path,
                        glob=f"**/*{ext}",
                        loader_cls=loader_cls,
                        show_progress=show_progress,
                        use_multithreading=use_multithreading
                    )
                    docs = loader.load()
                    if docs:
                        if data.get("_dev", False):
                            logger.debug("\033[32mLoaded %d documents with extension %s\033[0m", len(docs), ext)
                        data["_session"]["documents"].extend(docs)
                except Exception as e:
                    if data.get("_dev", False):
                        logger.error("\033[31mError loading documents with extension %s: %s\033[0m", ext, str(e))
                    continue

            if data.get("_dev", False):
                total_docs = len(data["_session"]["documents"])
                logger.debug("\033[32mTotal documents loaded: %d\033[0m", total_docs)

        except Exception as e:
            if data.get("_dev", False):
                logger.error("\033[31mError in directory_load: %s\033[0m", str(e))
            raise

        return data
    return RunnableLambda(inner)
