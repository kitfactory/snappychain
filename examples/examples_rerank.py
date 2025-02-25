"""
Example of using the UnifiedRerank for document reranking.
ドキュメント再ランキングのための UnifiedRerank の使用例。
"""

import sys
import os
import json

# srcディレクトリをパスに追加して、snappychainモジュールをインポートできるようにします
# Add src directory to path to import snappychain module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from snappychain.rerank import build_reranker
from langchain_core.documents import Document

# Cohereリランクは使用しないためコメントアウト
# Commenting out Cohere rerank example as we won't use it
"""
def example_cohere_rerank():
    '''
    Example of using Cohere for document reranking.
    Cohereを使用したドキュメント再ランキングの例。
    '''
    print("\n=== Cohereリランクの例 / Cohere Rerank Example ===")
    
    # サンプルドキュメントの作成
    # Create sample documents
    documents = [
        Document(page_content="Python is a high-level programming language known for its readability and versatility."),
        Document(page_content="JavaScript is widely used for web development and can run in browsers."),
        Document(page_content="Python has libraries like NumPy and Pandas that are popular for data analysis."),
        Document(page_content="Machine learning frameworks in Python include TensorFlow and PyTorch."),
        Document(page_content="React is a JavaScript library for building user interfaces."),
    ]
    
    # クエリ
    # Query
    query = "Python for data analysis"
    
    # Cohereリランカーの設定
    # Configure Cohere reranker
    config = {
        "provider": "cohere",
        "settings": {
            "model": "rerank-multilingual-v2.0",
            "top_n": 3
        }
    }
    
    try:
        # リランカーの構築
        # Build reranker
        reranker = build_reranker(config)
        
        # ドキュメントの再ランク
        # Rerank documents
        reranked_docs = reranker.compress_documents(documents, query)
        
        # 結果の表示
        # Display results
        print(f"\nクエリ / Query: {query}")
        print("\n再ランクされたドキュメント / Reranked documents:")
        for i, doc in enumerate(reranked_docs):
            print(f"{i+1}. {doc.page_content}")
            
        return reranked_docs
    except Exception as e:
        print(f"エラー / Error: {str(e)}")
        return None
"""

def example_llm_rerank():
    """
    Example of using LLM for document reranking.
    LLMを使用したドキュメント再ランキングの例。
    """
    print("\n=== LLMリランクの例 / LLM Rerank Example ===")
    
    # サンプルドキュメントの作成
    # Create sample documents
    documents = [
        Document(page_content="Python is a high-level programming language known for its readability and versatility."),
        Document(page_content="JavaScript is widely used for web development and can run in browsers."),
        Document(page_content="Python has libraries like NumPy and Pandas that are popular for data analysis."),
        Document(page_content="Machine learning frameworks in Python include TensorFlow and PyTorch."),
        Document(page_content="React is a JavaScript library for building user interfaces."),
    ]
    
    # クエリ
    # Query
    query = "Python for data analysis"
    
    # LLMリランカーの設定
    # Configure LLM reranker
    config = {
        "provider": "llm",
        "settings": {
            "model": "gpt-4o-mini",
            "top_n": 3
        },
        "debug": True
    }
    
    try:
        # リランカーの構築
        # Build reranker
        reranker = build_reranker(config)
        
        # ドキュメントの再ランク
        # Rerank documents
        reranked_docs = reranker.compress_documents(documents, query)
        
        # 結果の表示
        # Display results
        print(f"\nクエリ / Query: {query}")
        print("\n再ランクされたドキュメント / Reranked documents:")
        for i, doc in enumerate(reranked_docs):
            print(f"{i+1}. {doc.page_content}")
            if hasattr(doc, "metadata") and "rerank_explanation" in doc.metadata:
                print(f"   理由 / Reason: {doc.metadata['rerank_explanation']}")
            
        return reranked_docs
    except Exception as e:
        print(f"エラー / Error: {str(e)}")
        return None

if __name__ == "__main__":
    print("=== ドキュメント再ランキングの例 / Document Reranking Examples ===")
    
    # Cohereの例はCohereのAPIキーが必要なためコメントアウト済み
    # Cohere example is already commented out as it requires a Cohere API key
    # example_cohere_rerank()
    
    # LLMの例を実行
    # Run LLM example
    example_llm_rerank()
    
    print("\n=== 例が完了しました / Examples completed ===") 