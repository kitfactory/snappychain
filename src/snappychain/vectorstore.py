"""
Vector store operations module.
ベクトルストア操作モジュール。
"""

from typing import Optional, List
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.runnables import RunnableLambda
import os
from onelogger import Logger

logger = Logger.get_logger(__name__)

class UnifiedVectorStore():
    """
    UnifiedVectorStore class to manage vector store operations.
    ベクトルストアの操作を統一的に管理するクラス。
    
    Attributes:
        settings (dict): Configuration for vector store settings (e.g., provider, save_dir)
                           ベクトルストア設定用の辞書（例: プロバイダー、保存ディレクトリ）。
        provider (str): The provider type, e.g., 'faiss' or 'chroma'
                          プロバイダーの種類。例: 'faiss'または'chroma'。
        vector_store: The underlying vector store instance.
                      内部で使用されるベクトルストアのインスタンス。
        embeddings: The embeddings model used for processing documents.
                    文書処理に使用される埋め込みモデル。
    """
    
    def __init__(self, settings: dict[str, object], embeddings: Embeddings):
        """
        Initialize the UnifiedVectorStore with given settings and embeddings.
        与えられた設定と埋め込みモデルを使用してUnifiedVectorStoreを初期化します。
        
        Args:
            settings (dict): Vector store configuration settings (e.g., provider, save_dir)
                             ベクトルストア設定用の辞書（例: プロバイダー、保存ディレクトリ）。
            embeddings: An embeddings model instance for document processing.
                        文書処理に使用する埋め込みモデルのインスタンス。
        """
        if embeddings is None:
            raise ValueError("embeddings is required")
        self.embeddings = embeddings
        self.settings = settings
        self.provider = settings.get("provider", "faiss").lower()
        if self.provider == "faiss":
            self._faiss_setting()
        elif self.provider == "chroma":
            self._chroma_setting()
        else:
            raise ValueError(f"Unsupported vector store provider: {self.provider}")

    def _faiss_create_new(self):
        """
        Create a new FAISS vector store from scratch.
        新規にFAISSベクトルストアを作成します。
        
        This method initializes a new FAISS index and sets up an in-memory docstore.
        このメソッドは、新しいFAISSインデックスを初期化し、インメモリのdocstoreをセットアップします。
        """
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore

        index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))

        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
    
    def _faiss_load(self):
        """
        Load an existing FAISS vector store from the save directory.
        保存ディレクトリから既存のFAISSベクトルストアを読み込みます。
        """
        logger.info("\033[34mLoading FAISS from %s\033[0m", self.save_to)
        self.vector_store = FAISS.load_local(
            folder_path=self.settings.get("save_dir"),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
    
    def _faiss_setting(self):
        """
        Configure the FAISS vector store.
        FAISSベクトルストアを設定します。
        
        This method checks for the save directory in the settings and either creates a new FAISS index
        or loads an existing one.
        このメソッドは設定内の保存ディレクトリの有無を確認し、新規のFAISSインデックスを作成するか、既存のものをロードします。
        """
        self.save_to = self.settings.get("save_dir", None)
        if self.save_to is None:
            raise ValueError("save_dir is required in FAISS settings")

        import os
        if self.save_to and not os.path.exists(self.save_to):
            self._faiss_create_new()
        else:
            self._faiss_load()

    def _chroma_setting(self):
        """
        Configure the Chroma vector store.
        Chromaベクトルストアを設定します。
        
        This method initializes a new Chroma instance or loads an existing one.
        このメソッドは、新規のChromaインスタンスを初期化するか、既存のものをロードします。
        """
        self.save_to = self.settings.get("save_dir", None)
        if self.save_to is None:
            raise ValueError("save_dir is required in Chroma settings")
        
        import os
        os.makedirs(self.save_to, exist_ok=True)
        
        self.vector_store = Chroma(
            persist_directory=self.save_to,
            embedding_function=self.embeddings,
        )

    def add_documents(self, docs):
        """
        Add documents to the vector store and save the updated index.
        ドキュメントをベクトルストアに追加し、更新されたインデックスを保存します。
        
        Args:
            docs: A list of documents to add.
                  追加するドキュメントのリスト。
        """
        self.vector_store.add_documents(docs)
        if self.provider == "faiss":
            self.vector_store.save_local(self.settings.get("save_dir"))
        elif self.provider == "chroma":
            self.vector_store.persist()
    
    def similarity_search(self, query, k=1):
        """
        Perform a similarity search on the vector store.
        ベクトルストア上で類似性検索を実行します。
        
        Args:
            query (str): The query text.
                         検索用クエリのテキスト。
            k (int, optional): Number of top results to return (default is 1).
                               返す上位の結果数（デフォルトは1）。
        
        Returns:
            list: A list of matching documents.
                  一致したドキュメントのリスト。
        """
        results = self.vector_store.similarity_search(query, k=k)
        logger.debug("\033[34mVector search results: %s\033[0m", 
                    [f"{i+1}. {doc.page_content[:50]}..." for i, doc in enumerate(results)])
        return results

    def as_retriever(self, **kwargs):
        """
        Convert the vector store to a retriever.
        ベクトルストアをリトリーバーに変換します。

        Args:
            **kwargs: Additional arguments to pass to the underlying vector store's as_retriever method.
                     内部のベクトルストアのas_retrieverメソッドに渡す追加の引数。

        Returns:
            BaseRetriever: A retriever instance that can be used for document retrieval.
                          文書検索に使用できるリトリーバーインスタンス。

        Raises:
            ValueError: If the vector store is not initialized.
                       ベクトルストアが初期化されていない場合に発生します。
        """
        if not hasattr(self, 'vector_store'):
            raise ValueError("Vector store is not initialized")
        return self.vector_store.as_retriever(**kwargs)


def faiss_vectorstore(persist_dir: Optional[str] = None) -> RunnableLambda:
    """
    Create/load FAISS vector store and add documents.
    FAISSベクトルストアを作成/ロードし、ドキュメントを追加します。

    Args:
        persist_dir (Optional[str]): Directory to save/load the vector store.
                                    ベクトルストアを保存/ロードするディレクトリ。

    Returns:
        RunnableLambda: A runnable that creates or loads a FAISS vector store.
                       FAISSベクトルストアを作成またはロードするRunnable。
    """
    def _faiss_vectorstore(data):
        docs = data.get("documents", [])
        embeddings = data.get("embeddings", None)
        
        if embeddings is None:
            raise ValueError("embeddings is required")
            
        settings = {
            "provider": "faiss",
            "save_dir": persist_dir
        }
        
        vector_store = UnifiedVectorStore(settings=settings, embeddings=embeddings)
        if docs:
            vector_store.add_documents(docs)
            
        return vector_store
        
    return RunnableLambda(_faiss_vectorstore)
