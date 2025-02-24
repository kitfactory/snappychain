"""
Dummy vector store for analyzing method calls.
メソッド呼び出しを分析するためのダミーベクトルストア。
"""

from typing import Any, Dict, Iterable, List, Optional
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

class DummyVectorStore(VectorStore):
    """
    A dummy vector store that logs method calls.
    メソッド呼び出しを記録するダミーベクトルストア。
    """
    
    def __init__(self, embeddings: Optional[Embeddings] = None):
        """Initialize the dummy vector store.
        ダミーベクトルストアを初期化します。
        """
        self._embeddings = embeddings or OpenAIEmbeddings()
        self.docs: List[Document] = []
        print(f"\n[DummyVectorStore.__init__] Called with embeddings: {type(embeddings)}")

    @property
    def embeddings(self) -> Embeddings:
        """Get the embeddings instance.
        埋め込みのインスタンスを取得します。
        """
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value: Embeddings) -> None:
        """Set the embeddings instance.
        埋め込みのインスタンスを設定します。
        """
        self._embeddings = value

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "DummyVectorStore":
        """Create DummyVectorStore from texts.
        テキストからDummyVectorStoreを作成します。
        """
        print(f"\n[DummyVectorStore.from_texts] Called with:")
        print(f"- texts: {len(texts)} items")
        print(f"- embedding: {type(embedding)}")
        print(f"- metadatas: {metadatas}")
        print(f"- kwargs: {kwargs}")
        
        instance = cls(embeddings=embedding)
        instance.add_texts(texts, metadatas=metadatas, **kwargs)
        return instance
        
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the store.
        テキストをストアに追加します。
        """
        print(f"\n[DummyVectorStore.add_texts] Called with:")
        print(f"- texts: {len(list(texts))} items")
        print(f"- metadatas: {metadatas}")
        print(f"- kwargs: {kwargs}")
        return ["dummy_id"]

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to the store.
        ドキュメントをストアに追加します。
        """
        print(f"\n[DummyVectorStore.add_documents] Called with:")
        print(f"- documents: {len(documents)} items")
        print(f"- kwargs: {kwargs}")
        self.docs.extend(documents)
        return ["dummy_id"] * len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.
        クエリに最も類似したドキュメントを返します。
        """
        print(f"\n[DummyVectorStore.similarity_search] Called with:")
        print(f"- query: {query}")
        print(f"- k: {k}")
        print(f"- kwargs: {kwargs}")
        return self.docs[:k] if self.docs else []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[tuple[Document, float]]:
        """Return docs most similar to query, with scores.
        クエリに最も類似したドキュメントをスコア付きで返します。
        """
        print(f"\n[DummyVectorStore.similarity_search_with_score] Called with:")
        print(f"- query: {query}")
        print(f"- k: {k}")
        print(f"- kwargs: {kwargs}")
        return [(doc, 0.99) for doc in self.docs[:k]] if self.docs else []

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query (async).
        クエリに最も類似したドキュメントを返します（非同期）。
        """
        print(f"\n[DummyVectorStore.asimilarity_search] Called with:")
        print(f"- query: {query}")
        print(f"- k: {k}")
        print(f"- kwargs: {kwargs}")
        return self.docs[:k] if self.docs else []

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[tuple[Document, float]]:
        """Return docs most similar to query, with scores (async).
        クエリに最も類似したドキュメントをスコア付きで返します（非同期）。
        """
        print(f"\n[DummyVectorStore.asimilarity_search_with_score] Called with:")
        print(f"- query: {query}")
        print(f"- k: {k}")
        print(f"- kwargs: {kwargs}")
        return [(doc, 0.99) for doc in self.docs[:k]] if self.docs else []

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID.
        ベクトルIDで削除します。
        """
        print(f"\n[DummyVectorStore.delete] Called with:")
        print(f"- ids: {ids}")
        print(f"- kwargs: {kwargs}")
        return True

if __name__ == "__main__":
    # Create test documents
    # テストドキュメントの作成
    docs = [
        Document(page_content="This is a test document 1", metadata={"source": "test1"}),
        Document(page_content="This is a test document 2", metadata={"source": "test2"}),
        Document(page_content="This is a test document 3", metadata={"source": "test3"}),
    ]
    
    # Initialize vector store
    # ベクトルストアの初期化
    print("\n=== Test 1: Basic Vector Store Operations ===")
    vs = DummyVectorStore()
    
    # Add documents
    # ドキュメントの追加
    vs.add_documents(docs)
    
    # Similarity search
    # 類似度検索
    results = vs.similarity_search("test", k=2)
    
    # Test with RetrievalQA
    # RetrievalQAでのテスト
    print("\n=== Test 2: RetrievalQA Integration ===")
    
    # Initialize components
    # コンポーネントの初期化
    llm = ChatOpenAI(model="gpt-4o-mini")
    retriever = vs.as_retriever(search_kwargs={"k": 2})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    
    # Run query
    # クエリの実行
    result = qa.invoke("What is in the test documents?")