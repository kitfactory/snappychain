# bm25sj.py
# This module provides a BM25SJ Retriever implementation for Japanese text retrieval.
# このモジュールは日本語テキスト検索のためのBM25SJリトリーバーを提供します。

from typing import Dict, List, Any, Optional, Tuple, ClassVar
from langchain_core.runnables import RunnableLambda
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from onelogger import Logger
import re
import pickle
import os
import numpy as np
import bm25s
from pydantic import Field, PrivateAttr

logger = Logger.get_logger(__name__)

class BM25SJRetriever(BaseRetriever):
    """
    BM25SJ Retriever for Japanese text.
    日本語テキスト用のBM25SJリトリーバー。
    
    This is an implementation of BM25 algorithm optimized for Japanese language using bm25s-j,
    incorporating specialized tokenization for better performance with Japanese text.
    これはbm25s-jを使用した日本語向けに最適化されたBM25アルゴリズムの実装で、
    専用のトークン化を組み込んで日本語テキストのパフォーマンスを向上させています。
    """
    
    # pydanticのプライベート属性として実装
    _k1_param: float = PrivateAttr(default=1.5)
    _b_param: float = PrivateAttr(default=0.75)
    _corpus_texts: List = PrivateAttr(default_factory=list)
    _original_docs: Dict = PrivateAttr(default_factory=dict)
    _retriever: Any = PrivateAttr(default=None)
    
    # pydanticのパブリックフィールド
    k: int = Field(default=4, description="Number of documents to retrieve")
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def __init__(
        self, 
        documents: Optional[List[Document]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        k: int = 4,
        **kwargs
    ):
        """
        Initialize the BM25SJ Retriever.
        BM25SJリトリーバーを初期化します。
        
        Args:
            documents (Optional[List[Document]]): Documents to index.
                                                インデックスするドキュメント。
            k1 (float): BM25 parameter that controls term frequency saturation.
                       単語頻度の飽和を制御するBM25パラメータ。
            b (float): BM25 parameter that controls document length normalization.
                      文書の長さの正規化を制御するBM25パラメータ。
            k (int): Default number of documents to retrieve.
                    デフォルトで取得するドキュメントの数。
        """
        # BaseRetrieverの初期化
        super().__init__(k=k, **kwargs)
        
        # プライベート属性の設定
        self._k1_param = k1
        self._b_param = b
        self._corpus_texts = []
        self._original_docs = {}
        
        # Initialize bm25s retriever
        # bm25sリトリーバーを初期化
        self._retriever = bm25s.BM25(k1=self._k1_param, b=self._b_param)
        
        if documents:
            self.add_documents(documents)
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the retriever.
        リトリーバーにドキュメントを追加します。
        
        Args:
            documents (List[Document]): Documents to add.
                                       追加するドキュメント。
        """
        # Extract text from documents
        # ドキュメントからテキストを抽出
        start_idx = len(self._corpus_texts)
        corpus_texts = [doc.page_content for doc in documents]
        
        # Store original documents and texts
        # 元のドキュメントとテキストを保存
        for i, doc in enumerate(documents):
            self._original_docs[start_idx + i] = doc
        
        # Update corpus texts
        # コーパステキストを更新
        self._corpus_texts.extend(corpus_texts)
        
        # Tokenize all texts using bm25s tokenizer and index
        # bm25sトークナイザーを使用してすべてのテキストをトークン化してインデックス化
        corpus_tokens = bm25s.tokenize(self._corpus_texts, stopwords="japanese")
        
        # Index the tokenized corpus
        # トークン化したコーパスをインデックス化
        self._retriever.index(corpus_tokens)
        
        logger.info(f"Added {len(documents)} documents to BM25SJ retriever, total {len(self._corpus_texts)}")
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for the given query.
        指定されたクエリに関連するドキュメントを取得します。
        
        Args:
            query (str): Query text.
                        クエリテキスト。
                    
        Returns:
            List[Document]: List of relevant documents.
                          関連ドキュメントのリスト。
        """
        if not self._corpus_texts:
            logger.warning("No documents have been added to the retriever")
            return []
        
        try:
            # Tokenize query using bm25s tokenizer
            # bm25sトークナイザーを使用してクエリをトークン化
            query_tokens = bm25s.tokenize(query, stopwords="japanese")
            
            # Retrieve documents and scores
            # ドキュメントとスコアを取得
            doc_indices, scores = self._retriever.retrieve(
                query_tokens, 
                k=self.k
            )
            
            # Convert to flat list if needed
            # 必要に応じてフラットなリストに変換
            if hasattr(doc_indices, "shape") and len(doc_indices.shape) > 1:
                doc_indices = doc_indices[0]
                scores = scores[0]
            
            # Map back to original documents
            # 元のドキュメントにマッピング
            result_docs = []
            
            for i, doc_idx in enumerate(doc_indices):
                # Ensure doc_idx is an integer
                if isinstance(doc_idx, (str, np.ndarray)):
                    try:
                        doc_idx = int(doc_idx)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid document index type: {type(doc_idx)}")
                        continue
                
                if doc_idx in self._original_docs:
                    # Clone the document to avoid modifying the original
                    doc = Document(
                        page_content=self._original_docs[doc_idx].page_content,
                        metadata=dict(self._original_docs[doc_idx].metadata)
                    )
                    # Add score to metadata
                    doc.metadata["score"] = float(scores[i])
                    result_docs.append(doc)
                    logger.debug(f"Retrieved document {doc_idx} with score {scores[i]}")
                else:
                    logger.warning(f"Document index {doc_idx} not found in original documents (max index: {max(self._original_docs.keys()) if self._original_docs else -1})")
            
            logger.debug(f"Retrieved {len(result_docs)} documents for query: {query}")
            return result_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def save(self, file_path: str) -> None:
        """
        Save the retriever to a file.
        リトリーバーをファイルに保存します。
        
        Args:
            file_path (str): Path to save the retriever.
                           リトリーバーを保存するパス。
        """
        data = {
            'corpus_texts': self._corpus_texts,
            'original_docs': self._original_docs,
            'k1': self._k1_param,
            'b': self._b_param,
            'k': self.k
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved BM25SJ retriever to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'BM25SJRetriever':
        """
        Load a retriever from a file.
        ファイルからリトリーバーを読み込みます。
        
        Args:
            file_path (str): Path to load the retriever from.
                           リトリーバーを読み込むパス。
                           
        Returns:
            BM25SJRetriever: Loaded retriever.
                           読み込まれたリトリーバー。
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        retriever = cls(
            k1=data['k1'],
            b=data['b'],
            k=data.get('k', 4)  # Default to 4 if k was not saved
        )
        
        retriever._corpus_texts = data['corpus_texts']
        retriever._original_docs = data['original_docs']
        
        # Recreate bm25s retriever and index corpus
        # bm25sリトリーバーを再作成してコーパスをインデックス化
        retriever._retriever = bm25s.BM25(k1=retriever._k1_param, b=retriever._b_param)
        
        if retriever._corpus_texts:
            # Tokenize and index the corpus
            # コーパスをトークン化してインデックス化
            corpus_tokens = bm25s.tokenize(retriever._corpus_texts, stopwords="japanese")
            retriever._retriever.index(corpus_tokens)
        
        logger.info(f"Loaded BM25SJ retriever from {file_path}")
        return retriever
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        k1: float = 1.5,
        b: float = 0.75,
        k: int = 4
    ) -> 'BM25SJRetriever':
        """
        Create a BM25SJ retriever from a list of documents.
        ドキュメントのリストからBM25SJリトリーバーを作成します。
        
        Args:
            documents (List[Document]): Documents to index.
                                      インデックスするドキュメント。
            k1 (float): BM25 parameter that controls term frequency saturation.
                      単語頻度の飽和を制御するBM25パラメータ。
            b (float): BM25 parameter that controls document length normalization.
                     文書の長さの正規化を制御するBM25パラメータ。
            k (int): Default number of documents to retrieve.
                   デフォルトで取得するドキュメントの数。
                   
        Returns:
            BM25SJRetriever: A BM25SJ retriever initialized with the given documents.
                           指定されたドキュメントで初期化されたBM25SJリトリーバー。
        """
        return cls(documents=documents, k1=k1, b=b, k=k)


def bm25sj(k: int = 4) -> RunnableLambda:
    """
    Create a BM25SJ retriever for Japanese text documents.
    日本語テキストドキュメント用のBM25SJリトリーバーを作成します。
    
    Args:
        k (int): Number of documents to retrieve in each query.
               各クエリで取得するドキュメントの数。
               
    Returns:
        RunnableLambda: A lambda that adds the BM25SJ retriever to the session.
                       BM25SJリトリーバーをセッションに追加するラムダ。
    """
    def inner(data):
        if "_session" not in data:
            data["_session"] = {}
            
        session = data["_session"]
        documents = session.get("documents", [])
        
        if not documents:
            logger.warning("No documents found in session to create BM25SJ retriever")
            return data
        
        # Create BM25SJ retriever
        # BM25SJリトリーバーを作成
        retriever = BM25SJRetriever(documents=documents, k=k)
        
        # Store in session
        # セッションに保存
        session["retriever"] = retriever
        session["retriever_type"] = "bm25sj"
        
        logger.info(f"Created BM25SJ retriever with {len(documents)} documents")
        return data
    
    return RunnableLambda(inner)


def bm25sj_query(query: str, k: int = None) -> RunnableLambda:
    """
    Query the BM25SJ retriever with the given text.
    指定されたテキストでBM25SJリトリーバーにクエリを実行します。
    
    Args:
        query (str): Query text.
                    クエリテキスト。
        k (int, optional): Number of documents to retrieve. If None, uses the retriever's default.
                          取得するドキュメントの数。Noneの場合、リトリーバーのデフォルトを使用します。
               
    Returns:
        RunnableLambda: A lambda that queries the retriever and adds results to the session.
                       リトリーバーにクエリを実行し、結果をセッションに追加するラムダ。
    """
    def inner(data):
        if "_session" not in data:
            data["_session"] = {}
            
        session = data["_session"]
        retriever = session.get("retriever")
        
        if not retriever:
            logger.warning("No retriever found in session. Create one with bm25sj() first.")
            return data
        
        if session.get("retriever_type") != "bm25sj":
            logger.warning(f"Retriever in session is not BM25SJ but {session.get('retriever_type')}")
            return data
        
        # If k is specified, temporarily override the retriever's default k
        if k is not None:
            orig_k = retriever.k
            retriever.k = k
        
        # Retrieve relevant documents
        # 関連ドキュメントを取得
        relevant_docs = retriever.get_relevant_documents(query)
        
        # Restore original k if it was changed
        if k is not None:
            retriever.k = orig_k
        
        # Store in session
        # セッションに保存
        session["similar_documents"] = relevant_docs
        
        logger.info(f"Retrieved {len(relevant_docs)} documents for query: {query}")
        return data
    
    return RunnableLambda(inner) 