# -*- coding: utf-8 -*-
"""
Example script for testing embedding functions in SnappyChain.
スプリッター関数の動作を確認するためのサンプルスクリプトです。
"""

import os
from snappychain.embedding import openai_embedding
from langchain.schema import Document
import numpy as np


def create_sample_documents():
    """
    Create sample documents for testing embeddings.
    埋め込みテスト用のサンプルドキュメントを作成します。

    Returns:
        list: List of Document objects.
              Documentオブジェクトのリスト。
    """
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "素早い茶色の狐が怠けた犬を飛び越えます。",
        "Python is a versatile programming language.",
        "Pythonは汎用性の高いプログラミング言語です。",
        "Machine learning models process data efficiently.",
        "機械学習モデルはデータを効率的に処理します。"
    ]
    return [Document(page_content=text) for text in texts]


def test_openai_embedding():
    """
    Test OpenAI embedding function with sample documents.
    サンプルドキュメントを使用してOpenAIの埋め込み関数をテストします。
    """
    print("\n--- Testing OpenAI Embeddings ---")
    print("Creating sample documents...")
    
    # Create sample documents
    # サンプルドキュメントを作成
    docs = create_sample_documents()
    data = {"_session": {"documents": docs}}
    
    # Check if OPENAI_API_KEY is set
    # OPENAI_API_KEYが設定されているか確認
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key before running this script.")
        return
    
    try:
        print("\nGenerating embeddings...")
        # Generate embeddings
        # 埋め込みを生成
        result = openai_embedding().invoke(data)
        
        # Display results
        # 結果を表示
        print("\nEmbedding Results:")
        for idx, doc in enumerate(result["_session"]["documents"], 1):
            embedding = doc.metadata.get("embedding")
            if embedding is not None:
                print(f"\nDocument {idx}:")
                print(f"Text: {doc.page_content}")
                print(f"Embedding shape: {np.array(embedding).shape}")
                print(f"First 5 values: {np.array(embedding)[:5]}")
            else:
                print(f"\nDocument {idx}: No embedding generated")
    
    except Exception as e:
        print(f"Error during embedding generation: {str(e)}")


def main():
    """
    Main function to run the embedding example.
    埋め込み例を実行するメイン関数。
    """
    test_openai_embedding()


if __name__ == "__main__":
    main()
