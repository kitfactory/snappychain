# ステップバイステップRAG構築チュートリアル

このチュートリアルでは、SnappyChainを使って、ゼロから日本語対応のRAG（検索拡張生成）アプリケーションを構築する方法を詳しく解説します。

## 前提条件

このチュートリアルを進める前に、以下の準備が必要です：

1. Python 3.9以上のインストール
2. 仮想環境の作成と有効化
3. SnappyChainのインストール
4. OpenAI APIキーの取得（または使用するLLMのAPI情報）

## 目次

1. [プロジェクトのセットアップ](#1-プロジェクトのセットアップ)
2. [データの準備](#2-データの準備)
3. [基本的なRAGの構築](#3-基本的なragの構築)
4. [BM25SJリトリーバーの追加](#4-bm25sjリトリーバーの追加)
5. [リランカーの設定](#5-リランカーの設定)
6. [高度な機能：並列処理](#6-高度な機能並列処理)
7. [完成したアプリケーションの実行](#7-完成したアプリケーションの実行)

## 1. プロジェクトのセットアップ

まず、プロジェクトディレクトリを作成し、必要なパッケージをインストールします。

```bash
# プロジェクトディレクトリの作成
mkdir rag-tutorial
cd rag-tutorial

# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate  # Windowsの場合は .venv\Scripts\activate

# 必要なパッケージのインストール
pip install snappychain
```

環境変数にAPIキーを設定します：

```bash
# Linuxまたは
export OPENAI_API_KEY="your-api-key-here"

# Windows PowerShellの場合
$env:OPENAI_API_KEY="your-api-key-here"
```

## 2. データの準備

RAGシステムでは、質問に答えるための知識ベースとなるドキュメントが必要です。ここでは、サンプルデータを用意します。

`data.py`というファイルを作成し、以下のコードを追加します：

```python
from langchain.schema import Document

def load_sample_data():
    """サンプルデータのロード関数"""
    
    # 日本の都市に関するサンプルデータ
    cities_data = [
        Document(
            page_content="東京は日本の首都であり、世界で最も人口の多い都市の一つです。東京スカイツリーや東京タワーなどの観光名所があります。",
            metadata={"source": "cities_guide", "topic": "tokyo", "language": "ja"}
        ),
        Document(
            page_content="京都は日本の古都であり、金閣寺や清水寺など多くの寺院や神社があります。春の桜と秋の紅葉が特に美しいことで知られています。",
            metadata={"source": "cities_guide", "topic": "kyoto", "language": "ja"}
        ),
        Document(
            page_content="大阪は日本の第二の都市であり、たこ焼きやお好み焼きなどの食文化で有名です。大阪城や道頓堀などの観光スポットがあります。",
            metadata={"source": "cities_guide", "topic": "osaka", "language": "ja"}
        ),
        Document(
            page_content="横浜は東京の南に位置する港町で、みなとみらいの夜景や中華街が有名です。日本で最初の開港都市としての歴史があります。",
            metadata={"source": "cities_guide", "topic": "yokohama", "language": "ja"}
        ),
        Document(
            page_content="札幌は北海道の中心都市であり、雪祭りやスキーリゾートで知られています。ラーメンやジンギスカンなどの郷土料理が人気です。",
            metadata={"source": "cities_guide", "topic": "sapporo", "language": "ja"}
        )
    ]
    
    # 日本の観光地に関するサンプルデータ
    attractions_data = [
        Document(
            page_content="富士山は日本の最高峰であり、標高3,776メートルです。世界文化遺産に登録されており、多くの観光客が訪れます。",
            metadata={"source": "attractions_guide", "topic": "fuji", "language": "ja"}
        ),
        Document(
            page_content="宮島（厳島）は広島県にある島で、海に浮かぶ鳥居で有名な厳島神社があります。世界文化遺産に登録されています。",
            metadata={"source": "attractions_guide", "topic": "miyajima", "language": "ja"}
        ),
        Document(
            page_content="沖縄の美ら海水族館は、世界最大級の水槽を持つ水族館です。ジンベエザメなどの大型海洋生物を観察できます。",
            metadata={"source": "attractions_guide", "topic": "okinawa", "language": "ja"}
        )
    ]
    
    # 日本の食文化に関するサンプルデータ
    food_data = [
        Document(
            page_content="寿司は酢飯に魚介類や野菜を組み合わせた日本の伝統的な料理です。江戸時代から発展し、現在では世界中で人気があります。",
            metadata={"source": "food_guide", "topic": "sushi", "language": "ja"}
        ),
        Document(
            page_content="ラーメンは中国から伝わった麺料理ですが、日本で独自の発展を遂げました。地域ごとに特色ある味があり、札幌の味噌、博多の豚骨などが有名です。",
            metadata={"source": "food_guide", "topic": "ramen", "language": "ja"}
        ),
        Document(
            page_content="天ぷらは食材を衣をつけて油で揚げた日本料理です。さくっとした食感と素材の味わいを楽しむことができます。",
            metadata={"source": "food_guide", "topic": "tempura", "language": "ja"}
        )
    ]
    
    # すべてのデータを結合
    all_documents = cities_data + attractions_data + food_data
    
    return all_documents
```

## 3. 基本的なRAGの構築

次に、基本的なRAGシステムを構築します。`simple_rag.py`というファイルを作成して以下のコードを追加します：

```python
import os
from data import load_sample_data
from snappychain.rag import build_rag_chain

# データディレクトリの作成
os.makedirs("./data", exist_ok=True)
os.makedirs("./data/vectorstore", exist_ok=True)

# 基本的なRAG設定
config = {
    "llm": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.2
    },
    "embeddings": {
        "provider": "openai",
        "model": "text-embedding-3-small"
    },
    "vector_store": {
        "provider": "FAISS",
        "settings": {
            "persist_dir": "./data/vectorstore"
        }
    }
}

def build_simple_rag():
    """基本的なRAGシステムの構築"""
    
    # RAGインスタンスの構築
    rag = build_rag_chain(config)
    
    # サンプルデータのロード
    documents = load_sample_data()
    print(f"{len(documents)}個のドキュメントを読み込みました")
    
    # ドキュメントの追加
    rag.add_documents(documents)
    print("ドキュメントを追加しました")
    
    return rag

def simple_qa_demo():
    """基本的なQ&Aデモ"""
    
    rag = build_simple_rag()
    
    # いくつかのサンプルクエリで試してみる
    sample_queries = [
        "東京について教えてください",
        "京都の観光スポットは？",
        "日本の伝統的な食べ物を教えてください"
    ]
    
    for query in sample_queries:
        print(f"\n質問: {query}")
        response = rag.query(query)
        print(f"回答: {response}")
        print("-" * 50)

if __name__ == "__main__":
    simple_qa_demo()
```

このスクリプトを実行すると、基本的なRAGシステムが構築され、サンプルクエリに対する応答が表示されます。

```bash
python simple_rag.py
```

## 4. BM25SJリトリーバーの追加

次に、日本語に最適化されたBM25SJリトリーバーを追加します。`bm25sj_rag.py`というファイルを作成して以下のコードを追加します：

```python
import os
from data import load_sample_data
from snappychain.rag import build_rag_chain

# データディレクトリの作成
os.makedirs("./data", exist_ok=True)
os.makedirs("./data/vectorstore", exist_ok=True)
os.makedirs("./data/bm25store", exist_ok=True)

# BM25SJリトリーバーを含むRAG設定
config = {
    "llm": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.2
    },
    "embeddings": {
        "provider": "openai",
        "model": "text-embedding-3-small"
    },
    "vector_store": {
        "provider": "FAISS",
        "settings": {
            "persist_dir": "./data/vectorstore"
        }
    },
    "retrievers": [
        {
            "provider": "BM25SJ",
            "settings": {
                "k1": 1.2,
                "b": 0.75,
                "k": 10,
                "save_dir": "./data/bm25store"
            }
        }
    ]
}

def build_bm25sj_rag():
    """BM25SJリトリーバーを含むRAGシステムの構築"""
    
    # RAGインスタンスの構築
    rag = build_rag_chain(config)
    
    # サンプルデータのロード
    documents = load_sample_data()
    print(f"{len(documents)}個のドキュメントを読み込みました")
    
    # ドキュメントの追加
    rag.add_documents(documents)
    print("ドキュメントを追加しました")
    
    return rag

def bm25sj_qa_demo():
    """BM25SJを使用したQ&Aデモ"""
    
    rag = build_bm25sj_rag()
    
    # いくつかのサンプルクエリで試してみる
    sample_queries = [
        "東京について教えてください",
        "京都の観光スポットは？",
        "日本の伝統的な食べ物を教えてください",
        "富士山はどのくらいの高さですか？",
        "大阪の食文化について説明してください"
    ]
    
    for query in sample_queries:
        print(f"\n質問: {query}")
        response = rag.query(query)
        print(f"回答: {response}")
        print("-" * 50)

if __name__ == "__main__":
    bm25sj_qa_demo()
```

このスクリプトを実行すると、BM25SJリトリーバーを含むRAGシステムが構築され、サンプルクエリに対する応答が表示されます。

```bash
python bm25sj_rag.py
```

## 5. リランカーの設定

次に、検索結果をより適切に並べ替えるためのリランカーを追加します。`reranker_rag.py`というファイルを作成して以下のコードを追加します：

```python
import os
import time
from data import load_sample_data
from snappychain.rag import build_rag_chain

# データディレクトリの作成
os.makedirs("./data", exist_ok=True)
os.makedirs("./data/vectorstore", exist_ok=True)
os.makedirs("./data/bm25store", exist_ok=True)

# リランカーを含むRAG設定
config = {
    "llm": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.2
    },
    "embeddings": {
        "provider": "openai",
        "model": "text-embedding-3-small"
    },
    "vector_store": {
        "provider": "FAISS",
        "settings": {
            "persist_dir": "./data/vectorstore"
        }
    },
    "retrievers": [
        {
            "provider": "BM25SJ",
            "settings": {
                "k1": 1.2,
                "b": 0.75,
                "k": 10,
                "save_dir": "./data/bm25store"
            }
        }
    ],
    "reranker": {
        "provider": "llm",
        "llm_provider": "openai",
        "model": "gpt-4o-mini",  # より高性能なモデルを使用
        "temperature": 0.0
    }
}

def build_reranker_rag():
    """リランカーを含むRAGシステムの構築"""
    
    # RAGインスタンスの構築
    rag = build_rag_chain(config)
    
    # サンプルデータのロード
    documents = load_sample_data()
    print(f"{len(documents)}個のドキュメントを読み込みました")
    
    # ドキュメントの追加
    rag.add_documents(documents)
    print("ドキュメントを追加しました")
    
    return rag

def reranker_qa_demo():
    """リランカーを使用したQ&Aデモ"""
    
    rag = build_reranker_rag()
    
    # いくつかのサンプルクエリで試してみる
    sample_queries = [
        "京都と大阪の違いは何ですか？",
        "日本の観光地で最もおすすめはどこですか？",
        "日本の食文化の特徴を説明してください",
        "東京と横浜はどのような関係がありますか？"
    ]
    
    for query in sample_queries:
        print(f"\n質問: {query}")
        start_time = time.time()
        response = rag.query(query)
        query_time = time.time() - start_time
        print(f"回答（処理時間: {query_time:.2f}秒）:")
        print(response)
        print("-" * 50)

if __name__ == "__main__":
    reranker_qa_demo()
```

このスクリプトを実行すると、リランカーを含むRAGシステムが構築され、サンプルクエリに対する応答が表示されます。

```bash
python reranker_rag.py
```

## 6. 高度な機能：並列処理

次に、複数のクエリを効率的に処理するための並列処理機能を追加します。`parallel_rag.py`というファイルを作成して以下のコードを追加します：

```python
import os
import time
from data import load_sample_data
from snappychain.rag import build_rag_chain

# データディレクトリの作成
os.makedirs("./data", exist_ok=True)
os.makedirs("./data/vectorstore", exist_ok=True)
os.makedirs("./data/bm25store", exist_ok=True)

# 完全なRAG設定
config = {
    "llm": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.2
    },
    "embeddings": {
        "provider": "openai",
        "model": "text-embedding-3-small"
    },
    "vector_store": {
        "provider": "FAISS",
        "settings": {
            "persist_dir": "./data/vectorstore"
        }
    },
    "retrievers": [
        {
            "provider": "BM25SJ",
            "settings": {
                "k1": 1.2,
                "b": 0.75,
                "k": 10,
                "save_dir": "./data/bm25store"
            }
        }
    ],
    "reranker": {
        "provider": "llm",
        "llm_provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.0
    }
}

def build_parallel_rag():
    """並列処理機能を持つRAGシステムの構築"""
    
    # RAGインスタンスの構築
    rag = build_rag_chain(config)
    
    # サンプルデータのロード
    documents = load_sample_data()
    print(f"{len(documents)}個のドキュメントを読み込みました")
    
    # ドキュメントの追加
    rag.add_documents(documents)
    print("ドキュメントを追加しました")
    
    return rag

def parallel_qa_demo():
    """並列処理を使用したQ&Aデモ"""
    
    rag = build_parallel_rag()
    
    # 複数のクエリを定義
    queries = [
        "東京について教えてください",
        "京都の観光スポットは？",
        "大阪の食文化とは？",
        "富士山の高さは？",
        "日本の伝統的な食べ物は何ですか？"
    ]
    
    print(f"{len(queries)}個のクエリを並列処理します")
    
    # 逐次処理の時間計測
    print("\n===== 逐次処理 =====")
    sequential_start = time.time()
    sequential_results = []
    
    for query in queries:
        start_time = time.time()
        result = rag.query(query)
        query_time = time.time() - start_time
        sequential_results.append({"query": query, "result": result, "time": query_time})
    
    sequential_total = time.time() - sequential_start
    print(f"逐次処理の合計時間: {sequential_total:.2f}秒")
    
    # 並列処理の時間計測
    print("\n===== 並列処理 =====")
    parallel_start = time.time()
    
    # 並列クエリ処理（最大3スレッド）
    parallel_results = rag.batch_query(queries, max_workers=3)
    
    parallel_total = time.time() - parallel_start
    print(f"並列処理の合計時間: {parallel_total:.2f}秒")
    print(f"速度向上率: {sequential_total / parallel_total:.2f}倍")
    
    # 結果の表示
    print("\n===== 並列処理の結果 =====")
    for result in parallel_results:
        print(f"質問: {result['query']}")
        print(f"応答時間: {result['time']:.2f}秒")
        print(f"応答: {result['result']}")
        print("-" * 50)

if __name__ == "__main__":
    parallel_qa_demo()
```

このスクリプトを実行すると、並列処理機能を持つRAGシステムが構築され、複数のクエリを逐次処理と並列処理の両方で実行し、パフォーマンスの違いを比較します。

```bash
python parallel_rag.py
```

## 7. 完成したアプリケーションの実行

最後に、すべての機能を組み合わせた完全なRAGアプリケーションを作成します。`app.py`というファイルを作成して以下のコードを追加します：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SnappyChain RAGアプリケーションの完全な実装例
"""

import os
import time
from langchain.schema import Document
from snappychain.rag import build_rag_chain
from data import load_sample_data

# データディレクトリの作成
os.makedirs("./data", exist_ok=True)
os.makedirs("./data/vectorstore", exist_ok=True)
os.makedirs("./data/bm25store", exist_ok=True)

# 完全なRAG設定
config = {
    "llm": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.2
    },
    "embeddings": {
        "provider": "openai",
        "model": "text-embedding-3-small"
    },
    "vector_store": {
        "provider": "FAISS",
        "settings": {
            "persist_dir": "./data/vectorstore"
        }
    },
    "retrievers": [
        {
            "provider": "BM25SJ",
            "settings": {
                "k1": 1.2,
                "b": 0.75,
                "k": 10,
                "save_dir": "./data/bm25store"
            }
        }
    ],
    "reranker": {
        "provider": "llm",
        "llm_provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.0
    }
}

def initialize_rag():
    """RAGシステムの初期化"""
    print("RAGシステムを初期化しています...")
    
    # RAGインスタンスの構築
    rag = build_rag_chain(config)
    
    # サンプルデータのロード
    documents = load_sample_data()
    print(f"{len(documents)}個のドキュメントを読み込みました")
    
    # ドキュメントの追加
    rag.add_documents(documents)
    print("ドキュメントを追加しました")
    
    return rag

def interactive_qa(rag):
    """対話式Q&Aセッション"""
    print("\nSnappyChain RAGデモ（終了するには 'exit' と入力）")
    print("-" * 50)
    
    while True:
        query = input("\n質問を入力してください: ")
        if query.lower() in ["exit", "quit", "終了"]:
            break
            
        start_time = time.time()
        try:
            response = rag.query(query)
            query_time = time.time() - start_time
            print(f"\n回答（処理時間: {query_time:.2f}秒）:")
            print(response)
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")

def run_batch_queries(rag):
    """バッチクエリの実行"""
    print("\nバッチクエリのデモンストレーション")
    print("-" * 50)
    
    queries = [
        "東京の主要な観光スポットは？",
        "京都で訪れるべき神社仏閣は？",
        "日本の伝統的な食べ物について教えてください",
        "大阪の特徴的な食文化とは？",
        "富士山に登るにはどうしたらいいですか？"
    ]
    
    print(f"{len(queries)}個のクエリを並列処理します...")
    start_time = time.time()
    
    results = rag.batch_query(queries, max_workers=3)
    
    total_time = time.time() - start_time
    print(f"並列処理の合計時間: {total_time:.2f}秒")
    
    for result in results:
        print(f"\n質問: {result['query']}")
        print(f"応答時間: {result['time']:.2f}秒")
        print(f"応答: {result['result']}")
        print("-" * 50)

def add_new_documents(rag):
    """新しいドキュメントの追加"""
    print("\n新しいドキュメントの追加デモ")
    print("-" * 50)
    
    new_docs = [
        Document(
            page_content="北海道は日本最北の島で、広大な自然と新鮮な海産物で知られています。四季折々の景色が楽しめ、冬のスキーリゾートが特に人気です。",
            metadata={"source": "travel_guide", "topic": "hokkaido", "language": "ja"}
        ),
        Document(
            page_content="沖縄は日本最南端の県で、美しいビーチと独自の文化で知られています。年間を通して温暖な気候で、マリンスポーツやリゾートホテルが人気です。",
            metadata={"source": "travel_guide", "topic": "okinawa", "language": "ja"}
        )
    ]
    
    print(f"{len(new_docs)}個の新しいドキュメントを追加します")
    
    # ドキュメントの追加
    rag.add_documents(new_docs)
    
    # 新しいドキュメントに関するクエリ
    queries = [
        "北海道について教えてください",
        "沖縄の特徴は何ですか？"
    ]
    
    for query in queries:
        print(f"\n質問: {query}")
        response = rag.query(query)
        print(f"回答: {response}")
        print("-" * 50)

def main():
    """メイン関数"""
    # RAGシステムの初期化
    rag = initialize_rag()
    
    # 機能選択メニュー
    while True:
        print("\n===== SnappyChain RAGデモ =====")
        print("1. 対話式Q&A")
        print("2. バッチクエリのデモ")
        print("3. 新しいドキュメントの追加デモ")
        print("4. 終了")
        
        choice = input("選択してください (1-4): ")
        
        if choice == "1":
            interactive_qa(rag)
        elif choice == "2":
            run_batch_queries(rag)
        elif choice == "3":
            add_new_documents(rag)
        elif choice == "4":
            print("アプリケーションを終了します")
            break
        else:
            print("無効な選択です。1から4の数字を入力してください。")

if __name__ == "__main__":
    main()
```

このスクリプトを実行すると、完全なRAGアプリケーションが起動し、対話式Q&A、バッチクエリ処理、新しいドキュメントの追加などの機能を試すことができます。

```bash
python app.py
```

## まとめ

このチュートリアルでは、SnappyChainを使用して、ゼロから高度なRAGアプリケーションを構築する方法を学びました。主な機能として：

1. 基本的なRAGシステムの構築
2. 日本語に最適化されたBM25SJリトリーバーの追加
3. LLMベースのリランカーの設定
4. 並列クエリ処理による効率化
5. 動的なドキュメント追加と管理

これらの機能を組み合わせることで、日本語と英語の両方に対応した高性能なRAGアプリケーションを簡単に構築することができます。

## 次のステップ

- 独自のデータでRAGシステムを構築する
- カスタムリトリーバーの作成
- パフォーマンスのチューニング
- UIの追加（Streamlit、Gradioなど）

より詳細な情報は、SnappyChainの公式ドキュメントを参照してください。 