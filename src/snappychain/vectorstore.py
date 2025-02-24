from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS, Chroma
from typing import Optional, List, Dict
from langchain.schema import Document
import os


def faiss_vectorstore(persist_dir: Optional[str] = None) -> RunnableLambda:
    """
    Create/load FAISS vector store and add documents.
    FAISSベクトルストアを作成/ロードし、ドキュメントを追加します。

    Args:
        persist_dir (str): Directory to persist/store the FAISS index
                          FAISSインデックスの保存ディレクトリ

    Returns:
        RunnableLambda: A lambda that handles document storage
                        ドキュメント保存を処理するラムダ
    """
    def inner(data):
        docs = data.get("_session", {}).get("documents", [])
        if not docs:
            return data

        # Get embeddings model from session
        # セッションからembeddingsモデルを取得
        embeddings = data.get("_session", {}).get("embedding_model")
        if not embeddings:
            raise ValueError("No embeddings model found in session data. Please run openai_embedding_model() or ollama_embedding_model() first.")

        # Create or load vector store
        # ベクトルストアの作成/ロード
        if persist_dir and os.path.exists(os.path.join(persist_dir, "index.faiss")):
            vector_store = FAISS.load_local(
                folder_path=persist_dir,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            vector_store.add_documents(docs)
        else:
            vector_store = FAISS.from_documents(docs, embeddings)
        
        # Save if persist_dir specified
        # 保存ディレクトリが指定されている場合保存
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            vector_store.save_local(persist_dir)
        
        data["_session"]["vector_store"] = vector_store
        return data
    return RunnableLambda(inner)


def chroma_vectorstore(persist_dir: Optional[str] = None) -> RunnableLambda:
    """
    Create/load Chroma vector store and add documents.
    Chromaベクトルストアを作成/ロードし、ドキュメントを追加します。

    Args:
        persist_dir (str): Directory to persist/store the Chroma database
                          Chromaデータベースの保存ディレクトリ

    Returns:
        RunnableLambda: A lambda that handles document storage
                        ドキュメント保存を処理するラムダ
    """
    def inner(data):
        docs = data.get("_session", {}).get("documents", [])
        if not docs:
            return data

        # Get embeddings model from session
        # セッションからembeddingsモデルを取得
        embeddings = data.get("_session", {}).get("embedding_model")
        if not embeddings:
            raise ValueError("No embeddings model found in session data. Please run openai_embedding_model() or ollama_embedding_model() first.")

        # Create or load vector store
        # ベクトルストアの作成/ロード
        if persist_dir and os.path.exists(persist_dir):
            vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
            vector_store.add_documents(docs)
        else:
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_dir
            )
        
        data["_session"]["vector_store"] = vector_store
        return data
    return RunnableLambda(inner)
