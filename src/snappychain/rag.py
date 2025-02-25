"""
RAG (Retrieval Augmented Generation) implementation.
RAG（検索拡張生成）の実装。
"""

from typing import Dict, Any, Optional, List
from langchain_core.runnables import RunnableLambda
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain.chains import RetrievalQA
from onelogger import Logger
import os

logger = Logger.get_logger(__name__)

"""
{
    "embeddings": {
        "provider": "openai"|"ollama"
        "model": "model/name"
    },
    "vector_store": {
        "provider": "FAISS"|"Chroma",
        "save_dir": "path/to/persist/dir"
    },
    "llm": {
        "provider": "openai"|"ollama",
        "model": "model/name",
        "temperature": 0.2
    }
}
"""

class Rag():
    """
    RAG that combines vector store retrieval with LLM for enhanced responses.
    ベクトルストアの検索とLLMを組み合わせて高度な応答を生成するRAG。

    Attributes:
        config (Dict): Configuration for embeddings, vector store, and LLM
                      埋め込み、ベクトルストア、LLMの設定
        vector_store (Optional[FAISS|Chroma]): Vector store instance
                                             ベクトルストアのインスタンス
        embeddings (Optional[OpenAIEmbeddings|OllamaEmbeddings]): Embeddings instance
                                                                埋め込みのインスタンス
        llm (Optional[ChatOpenAI|ChatOllama]): LLM instance
                                         LLMのインスタンス
        qa_chain (Optional[RetrievalQA]): QA chain for retrieval and response
                                         検索と応答のためのQAチェーン
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAG with configuration.
        設定を使用してRAGを初期化します。

        Args:
            config (Dict[str, Any]): Configuration dictionary
                                    設定辞書
        """
        self.config = config
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.__post_init__()

    def __post_init__(self):
        """
        Initialize embeddings, vector store and LLM based on config.
        設定に基づいて埋め込み、ベクトルストア、LLMを初期化します。
        """
        # Initialize embeddings
        # 埋め込みの初期化
        try:
            if self.config["embeddings"]["provider"] == "openai":
                self.embeddings = OpenAIEmbeddings(
                    model=self.config["embeddings"]["model"]
                )
            elif self.config["embeddings"]["provider"] == "ollama":
                self.embeddings = OllamaEmbeddings(
                    model=self.config["embeddings"]["model"]
                )
            else:
                raise ValueError(f"Unsupported embeddings provider: {self.config['embeddings']['provider']}")
        except Exception as e:
            logger.error("\033[31mError initializing embeddings: %s\033[0m", str(e))
            raise

        # Initialize LLM
        # LLMの初期化
        try:
            if self.config["llm"]["provider"] == "openai":
                self.llm = ChatOpenAI(
                    model=self.config["llm"]["model"],
                    temperature=0.2
                )
            elif self.config["llm"]["provider"] == "ollama":
                self.llm = ChatOllama(
                    model=self.config["llm"]["model"]
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config['llm']['provider']}")
        except Exception as e:
            logger.error("\033[31mError initializing LLM: %s\033[0m", str(e))
            raise

        # Initialize vector store
        # ベクトルストアの初期化
        try:
            if not self.embeddings:
                raise ValueError("Embeddings must be initialized before vector store")

            persist_dir = self.config["vector_store"]["settings"].get("persist_dir")
            if not persist_dir:
                raise ValueError("persist_dir is required in vector_store settings")

            if self.config["vector_store"]["provider"].upper() == "FAISS":
                if persist_dir and os.path.exists(os.path.join(persist_dir, "index.faiss")):
                    self.vector_store = FAISS.load_local(
                        folder_path=persist_dir,
                        embeddings=self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                else:
                    logger.info("Initializing vector store with empty documents list")
                    # Initialize with empty documents list
                    self.vector_store = FAISS.from_texts(texts=["dummy text"],embedding=self.embeddings)
                    if persist_dir:
                        os.makedirs(persist_dir, exist_ok=True)
                        self.vector_store.save_local(persist_dir)

            elif self.config["vector_store"]["provider"].upper() == "CHROMA":
                if persist_dir:
                    os.makedirs(persist_dir, exist_ok=True)
                    self.vector_store = Chroma(
                        persist_directory=persist_dir,
                        embedding_function=self.embeddings
                    )
                else:
                    self.vector_store = Chroma(
                        embedding_function=self.embeddings
                    )
            else:
                raise ValueError(f"Unsupported vector store provider: {self.config['vector_store']['provider']}")

        except Exception as e:
            logger.error("\033[31mError initializing vector store: %s\033[0m", str(e))
            raise

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )

    def store_documents(self, documents: List[Document]) -> 'Rag':
        """
        Store documents in the vector store.
        ドキュメントをベクトルストアに保存します。

        Args:
            documents (List[Document]): List of documents to store
                                      保存するドキュメントのリスト

        Returns:
            Rag: Self for method chaining
                メソッドチェーン用の自身のインスタンス
        """
        if not documents:
            raise ValueError("No documents provided / ドキュメントが提供されていません")

        try:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()

        except Exception as e:
            logger.error("\033[31mError storing documents: %s\033[0m", str(e))
            raise

        return self

    def query(self, question: str) -> str:
        """
        Query the RAG with a question.
        RAGに質問を投げかけます。

        Args:
            question (str): Question to ask
                          質問内容

        Returns:
            str: Response from the RAG
                 RAGからの応答
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call store_documents first. / QAチェーンが初期化されていません。先にstore_documentsを呼び出してください。")

        try:
            # ベクトルストアから検索結果を取得
            results = self.vector_store.similarity_search(question)
            logger.debug("\033[34mVector search results: %s\033[0m", [doc.page_content for doc in results])
            
            return self.qa_chain.invoke(question)
        except Exception as e:
            logger.error("\033[31mError querying RAG: %s\033[0m", str(e))
            raise

def build_rag_chain(config: Dict[str, Any]) -> Rag:
    """
    Build a RAG from configuration.
    設定からRAGを構築します。

    Args:
        config (Dict[str, Any]): Configuration dictionary
                                設定辞書

    Returns:
        Rag: Initialized RAG
            初期化されたRAG
    """
    try:
        # Validate config
        # 設定の検証
        required_keys = ["embeddings", "vector_store", "llm"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key} / 必要な設定キーがありません: {key}")

        # Validate providers
        # プロバイダーの検証
        if config["embeddings"]["provider"] not in ["openai", "ollama"]:
            raise ValueError("Embeddings provider must be 'openai' or 'ollama' / 埋め込みプロバイダーは'openai'または'ollama'である必要があります")

        if config["vector_store"]["provider"].upper() not in ["FAISS", "CHROMA"]:
            raise ValueError("Vector store provider must be 'FAISS' or 'CHROMA' / ベクトルストアプロバイダーは'FAISS'または'CHROMA'である必要があります")

        if config["llm"]["provider"] not in ["openai", "ollama"]:
            raise ValueError("LLM provider must be 'openai' or 'ollama' / LLMプロバイダーは'openai'または'ollama'である必要があります")

        # Create RAG
        # RAGの作成
        return Rag(config=config)

    except Exception as e:
        logger.error("\033[31mError building RAG: %s\033[0m", str(e))
        raise
