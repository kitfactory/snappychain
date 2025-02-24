# -*- coding: utf-8 -*-
"""
Example script for testing FAISS vector store in SnappyChain.
FAISSベクトルストアの動作を確認するためのサンプルスクリプトです。
"""

import os
from langchain.schema import Document
from snappychain import (
    openai_embedding,
    add_documents_to_vector_store,
    persist_vector_store,
    query_vector_store
)
from snappychain.vectorstore import faiss_vectorstore


def create_sample_documents():
    """
    Create sample documents for testing vector store.
    ベクトルストアテスト用のサンプルドキュメントを作成します。

    Returns:
        list: List of Document objects
              Documentオブジェクトのリスト
    """
    texts = [
        "LangChainの基本概念について説明します",
        "機械学習モデルのトレーニング手順",
        "Pythonでのデータ前処理のベストプラクティス",
        "大規模言語モデルの応用例と課題",
        "自然言語処理における埋め込み技術の重要性"
    ]
    return [Document(page_content=text) for text in texts]


def test_faiss_vectorstore():
    """
    Test FAISS vector store functionality.
    FAISSベクトルストアの機能をテストします。
    """
    print("\n--- Testing FAISS Vector Store ---")
    
    # Create sample documents
    # サンプルドキュメントを作成
    docs = create_sample_documents()
    data = {"_session": {"documents": docs}}

    # Check OpenAI API key
    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return

    try:
        # Create pipeline
        # パイプラインを作成
        pipeline = (
            openai_embedding()
            | faiss_vectorstore()
            | add_documents_to_vector_store()
            | persist_vector_store("examples/vectorstores/faiss")
        )

        # Execute pipeline
        # パイプラインを実行
        print("\nCreating and persisting vector store...")
        result = pipeline.invoke(data)
        
        # Test similarity search
        # 類似性検索のテスト
        query = "機械学習の処理手順"
        print(f"\nSimilarity search for: {query}")
        
        search_pipeline = query_vector_store(query, k=2)
        result = search_pipeline.invoke(result)
        
        similar_docs = result["_session"]["similar_documents"]
        for idx, doc in enumerate(similar_docs, 1):
            print(f"\nResult {idx}:")
            print(f"Content: {doc.page_content}")

    except Exception as e:
        print(f"\nError during vector store operation: {str(e)}")


def main():
    """
    Main function to run the vector store example.
    ベクトルストア例を実行するメイン関数。
    """
    test_faiss_vectorstore()


if __name__ == "__main__":
    main()
