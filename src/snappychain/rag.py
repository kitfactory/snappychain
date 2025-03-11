"""
RAG (Retrieval Augmented Generation) implementation.
RAG（検索拡張生成）の実装。
"""

import os
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from concurrent.futures import ThreadPoolExecutor

from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from snappychain.print import verbose_print, debug_print, Color

from snappychain.bm25sj import BM25SJRetriever

"""
{
    "embeddings": {
        "provider": "openai"|"ollama"
        "model": "model/name"
    },
    "vector_store": {
        "provider": "FAISS"|"Chroma",
        "settings": {
            "persist_dir": "path/to/persist/dir"
        }
    },
    "llm": {
        "provider": "openai"|"ollama",
        "model": "model/name",
        "temperature": 0.2
    },
    "retrievers": [
        {
            "provider": "vectorstore"|"BM25SJ",
            "settings": {
                "k1": 1.2,
                "b": 0.75,
                "k": 10,
                "save_dir": "path/to/persist/dir"
            }
        }
    ],
    "reranker": {
        "provider": "llm",
        "model": "model/name",
        "temperature": 0.0
    }
}
"""

class Rag:
    """
    Modular RAG that combines multiple retrievers with LLM for enhanced responses.
    複数のリトリーバーとLLMを組み合わせて高度な応答を生成するモジュラーRAG。

    This implementation supports:
    - Multiple retrievers (vector store, BM25SJ, etc.)
    - Reranking of retrieved documents
    - Thread-safe document updates with exclusive control
    - Parallel query processing

    Attributes:
        config (Dict): Configuration for embeddings, vector store, LLM, retrievers, and reranker
                     埋め込み、ベクトルストア、LLM、リトリーバー、リランカーの設定
        vector_store (Optional[VectorStore]): Vector store instance
                                           ベクトルストアのインスタンス
        embeddings (Optional[Embeddings]): Embeddings instance
                                        埋め込みのインスタンス
        llm (Optional[BaseLanguageModel]): LLM instance
                                       LLMのインスタンス
        retrievers (List[BaseRetriever]): List of retriever instances
                                        リトリーバーインスタンスのリスト
        reranker (Optional[Callable]): Function to rerank documents
                                     ドキュメントを再ランク付けする関数
        write_lock (threading.Lock): Lock for thread-safe document updates
                                    スレッドセーフなドキュメント更新のためのロック
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
        self.retrievers = []
        self.reranker = None
        self.write_lock = threading.Lock()
        self.__post_init__()

    def __post_init__(self):
        """
        Initialize embeddings, vector store, retrievers, reranker and LLM based on config.
        設定に基づいて埋め込み、ベクトルストア、リトリーバー、リランカー、LLMを初期化します。
        """
        # Initialize embeddings
        # 埋め込みの初期化
        try:
            verbose_print("RAG", "Initializing embeddings...", Color.YELLOW)
            if self.config.get("embeddings"):
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
                debug_print("RAG", f"Initialized {self.config['embeddings']['provider']} embeddings with model {self.config['embeddings']['model']}", Color.GREEN)
        except Exception as e:
            verbose_print("RAG", f"Error initializing embeddings: {str(e)}", Color.RED)
            raise

        # Initialize LLM
        # LLMの初期化
        try:
            verbose_print("RAG", "Initializing LLM...", Color.YELLOW)
            if self.config["llm"]["provider"] == "openai":
                self.llm = ChatOpenAI(
                    model=self.config["llm"]["model"],
                    temperature=self.config["llm"].get("temperature", 0.2)
                )
            elif self.config["llm"]["provider"] == "ollama":
                self.llm = ChatOllama(
                    model=self.config["llm"]["model"],
                    temperature=self.config["llm"].get("temperature", 0.2)
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config['llm']['provider']}")
            debug_print("RAG", f"Initialized {self.config['llm']['provider']} LLM with model {self.config['llm']['model']}", Color.GREEN)
        except Exception as e:
            verbose_print("RAG", f"Error initializing LLM: {str(e)}", Color.RED)
            raise

        # Initialize vector store if specified
        # ベクトルストアが指定されている場合は初期化
        try:
            if self.config.get("vector_store"):
                verbose_print("RAG", "Initializing vector store...", Color.YELLOW)
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
                        debug_print("RAG", f"Loaded existing FAISS index from {persist_dir}", Color.GREEN)
                    else:
                        verbose_print("RAG", "Initializing vector store with empty documents list", Color.YELLOW)
                        # Initialize with empty documents list
                        self.vector_store = FAISS.from_texts(texts=["dummy text"], embedding=self.embeddings)
                        if persist_dir:
                            os.makedirs(persist_dir, exist_ok=True)
                            self.vector_store.save_local(persist_dir)
                            debug_print("RAG", f"Created new FAISS index in {persist_dir}", Color.GREEN)

                elif self.config["vector_store"]["provider"].upper() == "CHROMA":
                    if persist_dir:
                        os.makedirs(persist_dir, exist_ok=True)
                        self.vector_store = Chroma(
                            persist_directory=persist_dir,
                            embedding_function=self.embeddings
                        )
                        debug_print("RAG", f"Initialized Chroma with persist directory {persist_dir}", Color.GREEN)
                    else:
                        self.vector_store = Chroma(
                            embedding_function=self.embeddings
                        )
                        debug_print("RAG", "Initialized in-memory Chroma instance", Color.GREEN)
                else:
                    raise ValueError(f"Unsupported vector store provider: {self.config['vector_store']['provider']}")

        except Exception as e:
            verbose_print("RAG", f"Error initializing vector store: {str(e)}", Color.RED)
            raise

        # Initialize retrievers
        # リトリーバーの初期化
        try:
            verbose_print("RAG", "Initializing retrievers...", Color.YELLOW)
            # Add vector store retriever if available
            if self.vector_store:
                self.retrievers.append(self.vector_store.as_retriever())
                debug_print("RAG", "Added vector store retriever", Color.GREEN)
            
            # Add other retrievers from config
            if self.config.get("retrievers"):
                for retriever_config in self.config["retrievers"]:
                    if retriever_config["provider"] == "BM25SJ":
                        bm25_retriever = BM25SJRetriever(
                            k1=retriever_config["settings"].get("k1", 1.2),
                            b=retriever_config["settings"].get("b", 0.75),
                            k=retriever_config["settings"].get("k", 10),
                            save_dir=retriever_config["settings"].get("save_dir")
                        )
                        self.retrievers.append(bm25_retriever)
                        debug_print("RAG", "Added BM25SJ retriever", Color.GREEN)
                    # Add more retriever types as needed
            
            if not self.retrievers:
                verbose_print("RAG", "No retrievers configured. RAG will not be able to retrieve documents.", Color.YELLOW)
        
        except Exception as e:
            verbose_print("RAG", f"Error initializing retrievers: {str(e)}", Color.RED)
            raise

        # Initialize reranker if specified
        # リランカーが指定されている場合は初期化
        try:
            if self.config.get("reranker"):
                verbose_print("RAG", "Initializing reranker...", Color.YELLOW)
                if self.config["reranker"]["provider"] == "llm":
                    # Create LLM-based reranker
                    rerank_llm = None
                    if self.config["reranker"].get("model"):
                        # 修正: llm_providerキーを確認するか、デフォルトはopenaiとする
                        llm_provider = self.config["reranker"].get("llm_provider", "openai")
                        
                        if llm_provider == "openai":
                            rerank_llm = ChatOpenAI(
                                model=self.config["reranker"]["model"],
                                temperature=self.config["reranker"].get("temperature", 0.0)
                            )
                        elif llm_provider == "ollama":
                            rerank_llm = ChatOllama(
                                model=self.config["reranker"]["model"]
                            )
                        else:
                            verbose_print("RAG", f"Unsupported LLM provider: {llm_provider}, using default LLM", Color.YELLOW)
                            rerank_llm = self.llm
                    else:
                        # Use the same LLM as for the RAG
                        rerank_llm = self.llm
                    
                    if rerank_llm:
                        debug_print("RAG", "Initialized reranker with LLM", Color.GREEN)
                        
        except Exception as e:
            verbose_print("RAG", f"Error initializing reranker: {str(e)}", Color.RED)
            raise

    def add_documents(self, documents: List[Document]) -> 'Rag':
        """
        Add documents to all retrievers with thread-safe exclusive control.
        すべてのリトリーバーにドキュメントをスレッドセーフに追加します。

        Args:
            documents (List[Document]): List of documents to add
                                      追加するドキュメントのリスト

        Returns:
            Rag: Self for method chaining
                メソッドチェーン用の自身のインスタンス
        """
        if not documents:
            verbose_print("RAG", "No documents provided for addition", Color.YELLOW)
            return self

        # Use lock to ensure thread safety for document updates
        with self.write_lock:
            try:
                # Add to vector store if available
                if self.vector_store:
                    verbose_print("RAG", f"Adding {len(documents)} documents to vector store", Color.YELLOW)
                    self.vector_store.add_documents(documents)
                    if hasattr(self.vector_store, 'persist'):
                        self.vector_store.persist()
                
                # Add to other retrievers
                for retriever in self.retrievers:
                    if retriever and hasattr(retriever, 'add_documents') and retriever != self.vector_store.as_retriever():
                        verbose_print("RAG", f"Adding {len(documents)} documents to {retriever.__class__.__name__}", Color.YELLOW)
                        retriever.add_documents(documents)
                
                verbose_print("RAG", f"Successfully added {len(documents)} documents to all retrievers", Color.GREEN)
            
            except Exception as e:
                verbose_print("RAG", f"Error adding documents: {str(e)}", Color.RED)
                raise
        
        return self

    def store_documents(self, documents: List[Document]) -> 'Rag':
        """
        Legacy method for compatibility. Calls add_documents.
        互換性のための従来のメソッド。add_documentsを呼び出します。

        Args:
            documents (List[Document]): List of documents to store
                                      保存するドキュメントのリスト

        Returns:
            Rag: Self for method chaining
                メソッドチェーン用の自身のインスタンス
        """
        return self.add_documents(documents)

    def query(self, question: str, top_k: int = 4) -> str:
        """
        Query the RAG with a question and get a response.
        RAGに質問を投げかけて応答を取得します。

        Args:
            question (str): Question to ask
                          質問内容
            top_k (int): Number of documents to retrieve from each retriever
                        各リトリーバーから取得するドキュメントの数

        Returns:
            str: Response from the RAG
                 RAGからの応答
        """
        if not self.llm:
            raise ValueError("LLM not initialized / LLMが初期化されていません")
        
        if not self.retrievers:
            raise ValueError("No retrievers available / 利用可能なリトリーバーがありません")
        
        try:
            all_docs = []
            
            # Retrieve documents from all retrievers
            for retriever in self.retrievers:
                try:
                    docs = retriever.get_relevant_documents(question)
                    verbose_print("RAG", f"Retrieved {len(docs)} documents from {retriever.__class__.__name__}", Color.YELLOW)
                    all_docs.extend(docs)
                except Exception as e:
                    verbose_print("RAG", f"Error retrieving from {retriever.__class__.__name__}: {str(e)}", Color.RED)
            
            verbose_print("RAG", f"Retrieved total of {len(all_docs)} documents", Color.GREEN)
            
            # Rerank documents if reranker is available
            if self.reranker and all_docs:
                try:
                    all_docs = self.reranker(question, all_docs)
                    verbose_print("RAG", "Documents reranked successfully", Color.GREEN)
                except Exception as e:
                    verbose_print("RAG", f"Error during reranking: {str(e)}", Color.RED)
            
            # Limit to top_k*2 documents for the final prompt
            all_docs = all_docs[:top_k*2]
            
            # Create prompt template
            template = """
            以下の情報を利用して、ユーザーの質問に専門家として回答してください。

            情報:
            {context}

            質問: {question}

            回答:
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # Create context from documents
            context = "\n\n".join([f"文書 {i+1}:\n{doc.page_content}" for i, doc in enumerate(all_docs)])
            
            # Create the chain
            chain = prompt | self.llm | StrOutputParser()
            
            # Generate answer
            response = chain.invoke({"context": context, "question": question})
            return response
            
        except Exception as e:
            verbose_print("RAG", f"Error during RAG query: {str(e)}", Color.RED)
            raise

    def batch_query(self, questions: List[str], max_workers: int = 3) -> List[Dict[str, Any]]:
        """
        Process multiple queries in parallel.
        複数のクエリを並列に処理します。

        Args:
            questions (List[str]): List of questions to ask
                                 質問のリスト
            max_workers (int): Maximum number of parallel threads
                             並列スレッドの最大数

        Returns:
            List[Dict[str, Any]]: List of results with query, result, and time
                                結果、クエリ、時間を含む結果のリスト
        """
        if not questions:
            return []
        
        results = []
        
        def process_query(q):
            import time
            start_time = time.time()
            try:
                result = self.query(q)
                query_time = time.time() - start_time
                return {"query": q, "result": result, "time": query_time, "success": True}
            except Exception as e:
                verbose_print("RAG", f"Error processing query '{q}': {str(e)}", Color.RED)
                return {"query": q, "result": str(e), "time": time.time() - start_time, "success": False}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_query, questions))
        
        return results

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
        required_keys = ["llm"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key} / 必要な設定キーがありません: {key}")

        # Create RAG instance
        rag = Rag(config)
        return rag
        
    except Exception as e:
        verbose_print("RAG", f"Error building RAG chain: {str(e)}", Color.RED)
        raise
