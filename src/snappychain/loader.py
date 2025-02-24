from langchain.document_loaders import TextLoader  # Import LangChain's TextLoader // LangChainのTextLoaderをインポートします
from typing import List  # Import List type // List型をインポートします
from langchain_core.runnables import RunnableLambda  # Import RunnableLambda // RunnableLambdaをインポートします
from langchain.document_loaders import PyPDFLoader

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
