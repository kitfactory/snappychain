# example_bm25.py
# Example demonstrating the use of BM25SJ retriever for Japanese text retrieval.
# 日本語テキスト検索のためのBM25SJリトリーバーの使用例を示す。

import os
import sys
import logging
from typing import List, Optional
from pathlib import Path

# Add the src directory to path to import snappychain
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from langchain.schema import Document
from onelogger import Logger
from src.snappychain import BM25SJRetriever

logger = Logger.get_logger(__name__)
logger.setLevel(logging.INFO)

def create_sample_documents() -> List[Document]:
    """
    Create sample documents for demonstration.
    デモンストレーション用のサンプルドキュメントを作成します。
    
    Returns:
        List[Document]: Sample documents.
                        サンプルドキュメント。
    """
    return [
        Document(
            page_content="東京は日本の首都であり、世界最大の都市圏を形成しています。",
            metadata={"source": "tokyo_info", "id": 1}
        ),
        Document(
            page_content="富士山は日本の象徴的な山で、標高は3,776メートルです。",
            metadata={"source": "fujisan_info", "id": 2}
        ),
        Document(
            page_content="京都は794年から1868年まで日本の首都であり、多くの伝統的な寺院や神社があります。",
            metadata={"source": "kyoto_info", "id": 3}
        ),
        Document(
            page_content="大阪は日本第二の都市であり、関西地方最大の商業都市です。",
            metadata={"source": "osaka_info", "id": 4}
        ),
        Document(
            page_content="札幌は北海道の中心都市で、雪まつりが有名です。",
            metadata={"source": "sapporo_info", "id": 5}
        ),
        Document(
            page_content="沖縄は日本最南端の県で、独自の文化や美しいビーチがあります。",
            metadata={"source": "okinawa_info", "id": 6}
        ),
        Document(
            page_content="新幹線は日本の高速鉄道システムで、東京から大阪まで約2時間半で移動できます。",
            metadata={"source": "shinkansen_info", "id": 7}
        ),
        Document(
            page_content="浅草寺は東京都台東区にある寺院で、雷門が有名です。",
            metadata={"source": "asakusa_info", "id": 8}
        ),
        Document(
            page_content="築地市場は以前、世界最大の魚市場でしたが、2018年に豊洲市場に移転しました。",
            metadata={"source": "tsukiji_info", "id": 9}
        ),
        Document(
            page_content="日本の伝統的な料理には寿司、刺身、天ぷら、そば、うどんなどがあります。",
            metadata={"source": "japanese_food", "id": 10}
        )
    ]

def print_documents(docs: List[Document], title: str = "Documents") -> None:
    """
    Print documents in a formatted way.
    ドキュメントを整形して表示します。
    
    Args:
        docs (List[Document]): Documents to print.
                              表示するドキュメント。
        title (str): Title for the document list.
                    ドキュメントリストのタイトル。
    """
    print(f"\n===== {title} =====")
    for i, doc in enumerate(docs):
        print(f"[{i+1}] Source: {doc.metadata.get('source', 'unknown')}")
        score = doc.metadata.get('score', 'N/A')
        if score != 'N/A':
            score = f"{score:.4f}"
        print(f"    Score: {score}")
        print(f"    Content: {doc.page_content}")
        print("-" * 50)

def demonstrate_direct_usage() -> None:
    """
    Demonstrate direct usage of BM25SJRetriever class.
    BM25SJRetrieverクラスの直接的な使用法を示します。
    """
    print("\n🔍 BM25SJRetriever デモ開始")
    print("=" * 70)
    
    # Create sample documents
    # サンプルドキュメントを作成
    docs = create_sample_documents()
    print(f"作成したサンプルドキュメント数: {len(docs)}")
    
    # Create a BM25SJRetriever instance
    # BM25SJRetrieverインスタンスを作成
    retriever = BM25SJRetriever(documents=docs, k=3)
    print("BM25SJRetrieverを初期化しました")
    
    # Run a query
    # クエリを実行
    query = "東京にある有名な観光地"
    print(f"\nクエリ: '{query}'")
    results = retriever.get_relevant_documents(query)
    print_documents(results, "検索結果")
    
    # Run another query
    # 別のクエリを実行
    query = "日本の伝統的な食べ物"
    print(f"\nクエリ: '{query}'")
    results = retriever.get_relevant_documents(query)
    print_documents(results, "検索結果")
    
    # Add new documents and run query again
    # 新しいドキュメントを追加して再度クエリを実行
    new_docs = [
        Document(
            page_content="天ぷらは日本の代表的な料理で、魚介類や野菜にころもをつけて油で揚げます。",
            metadata={"source": "tempura_info", "id": 11}
        ),
        Document(
            page_content="寿司は酢飯に魚や海産物をのせた日本料理で、世界中で人気があります。",
            metadata={"source": "sushi_info", "id": 12}
        )
    ]
    
    print("\n新しいドキュメントを追加します")
    retriever.add_documents(new_docs)
    
    query = "日本の伝統的な食べ物"
    print(f"\n同じクエリを再実行: '{query}'")
    results = retriever.get_relevant_documents(query)
    print_documents(results, "更新後の検索結果")
    
    # Testing with different k value
    # 異なるk値でテスト
    print("\nk=5で検索結果を取得します")
    retriever.k = 5
    results = retriever.get_relevant_documents(query)
    print_documents(results, "k=5での検索結果")
    
    # Test saving and loading
    # 保存と読み込みをテスト
    print("\nリトリーバーを保存します")
    save_path = "bm25sj_test.pkl"
    retriever.save(save_path)
    
    print("保存したリトリーバーを読み込みます")
    loaded_retriever = BM25SJRetriever.load(save_path)
    
    query = "日本の食べ物"
    print(f"\n読み込んだリトリーバーでクエリ実行: '{query}'")
    results = loaded_retriever.get_relevant_documents(query)
    print_documents(results, "読み込み後の検索結果")
    
    # Clean up
    # クリーンアップ
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"\n一時ファイル {save_path} を削除しました")
    
    print("\n🏁 BM25SJRetriever デモ終了")
    print("=" * 70)

def demonstrate_from_documents() -> None:
    """
    Demonstrate creating a BM25SJRetriever using the from_documents class method.
    from_documentsクラスメソッドを使用したBM25SJRetrieverの作成方法を示します。
    """
    print("\n📚 BM25SJRetriever.from_documents デモ開始")
    print("=" * 70)
    
    # Create sample documents
    # サンプルドキュメントを作成
    docs = create_sample_documents()
    
    # Use from_documents class method
    # from_documentsクラスメソッドを使用
    retriever = BM25SJRetriever.from_documents(
        documents=docs,
        k1=1.2,  # Custom k1 parameter
        b=0.75,
        k=4      # Retrieve top 4 results
    )
    
    print("BM25SJRetriever.from_documents()で初期化しました")
    
    # Run a query
    # クエリを実行
    query = "日本の都市"
    print(f"\nクエリ: '{query}'")
    results = retriever.get_relevant_documents(query)
    print_documents(results, "検索結果")
    
    print("\n🏁 from_documents デモ終了")
    print("=" * 70)

def main() -> None:
    """
    Main function to run the demos.
    デモを実行するメイン関数。
    """
    # Clean up any existing test files before starting
    # 開始前に既存のテストファイルをクリーンアップ
    save_path = "bm25sj_test.pkl"
    if os.path.exists(save_path):
        os.remove(save_path)
        logger.info(f"既存の {save_path} ファイルを削除しました")
    
    demonstrate_direct_usage()
    demonstrate_from_documents()

if __name__ == "__main__":
    main() 