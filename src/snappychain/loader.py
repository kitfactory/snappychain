from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader
)
from typing import Optional, List
import os
from snappychain.print import verbose_print, debug_print, Color

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
            try:
                verbose_print("Loader", f"Loading text file: {file_path}", Color.YELLOW)
                loader = TextLoader(file_path, encoding=encoding)  # Initialize TextLoader with the file path and encoding // ファイルパスとエンコーディングでTextLoaderを初期化します
                documents = loader.load()       # Load the documents // ドキュメントを読み込みます
                data["_session"]["documents"].extend(documents)  # Append loaded documents to the session's documents list // 読み込んだドキュメントをセッションのリストに追加します
                debug_print("Loader", f"Loaded {len(documents)} documents from {file_path}", Color.GREEN)
            except Exception as e:
                verbose_print("Loader", f"Error loading text file {file_path}: {str(e)}", Color.RED)
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
            try:
                verbose_print("Loader", f"Loading PDF file: {file_path}", Color.YELLOW)
                loader = PyPDFLoader(file_path)  # Instantiate PyPDFLoader / PyPDFLoaderのインスタンスを生成
                documents = loader.load()      # Load the PDF document(s) / PDFドキュメントを読み込む
                data["_session"]["documents"].extend(documents)  # Append loaded documents to the session's documents list // 読み込んだドキュメントをセッションのリストに追加します
                debug_print("Loader", f"Loaded {len(documents)} pages from PDF {file_path}", Color.GREEN)
            except Exception as e:
                verbose_print("Loader", f"Error loading PDF file {file_path}: {str(e)}", Color.RED)
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
            try:
                verbose_print("Loader", f"Loading markdown file: {file_path}", Color.YELLOW)
                result = md.convert(file_path)  # Convert file using MarkItDown / MarkItDownを使用してファイルを変換
                # Create a Document with the converted text
                doc = Document(page_content=result.text_content, metadata={"source": file_path})
                data["_session"]["documents"].append(doc)  # Append the Document to the documents list
                debug_print("Loader", f"Loaded and converted markdown file {file_path}", Color.GREEN)
            except Exception as e:
                verbose_print("Loader", f"Error loading markdown file {file_path}: {str(e)}", Color.RED)
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
                verbose_print("Loader", f"Loading unstructured markdown file: {file_path}", Color.YELLOW)
                loader = UnstructuredMarkdownLoader(file_path)
                documents = loader.load()
                data["_session"]["documents"].extend(documents)
                debug_print("Loader", f"Loaded {len(documents)} documents from {file_path}", Color.GREEN)
            except Exception as e:
                verbose_print("Loader", f"Error loading markdown file {file_path}: {str(e)}", Color.RED)
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
            verbose_print("Loader", f"Loading files from directory: {directory_path}", Color.YELLOW)
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
                        data["_session"]["documents"].extend(docs)
                        debug_print("Loader", f"Loaded {len(docs)} documents with extension {ext}", Color.GREEN)
                except Exception as e:
                    verbose_print("Loader", f"Error loading documents with extension {ext}: {str(e)}", Color.RED)
                    continue

            debug_print("Loader", f"Total documents loaded: {len(data['_session']['documents'])}", Color.GREEN)

        except Exception as e:
            verbose_print("Loader", f"Error in directory_load: {str(e)}", Color.RED)
            raise

        return data
    return RunnableLambda(inner)
