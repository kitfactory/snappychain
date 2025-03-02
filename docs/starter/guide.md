# SnappyChain 入門ガイド

## はじめに

SnappyChainは、高度なLLM操作やRAG自然言語処理機能を簡単に構築できるPythonライブラリです。特に日本語と英語の両方に対応したRAG（検索拡張生成）機能を提供し、より正確で根拠に基づいた応答を生成することができます。

本ガイドでは、SnappyChainのインストール方法から基本的な使い方、そして高度なRAGシステムの構築までをステップバイステップで解説します。

## 特徴

SnappyChainの主な特徴は以下の通りです。LangChainで必要とされるソースコードを大きく減少させることが可能です。

- **簡単なチェーン記述** : 直感的にLangChainのチェーン(LCEL)を書くことが可能
- **統一インターフェース** : VectorStoreなどは簡単化して、統一インターフェースを提供しています。
- **高度なRAG**: BM25による検索、Rerankを含むRAGに対応

### 特徴1. 簡単・直感的なチェイン構築

__従来のLangChain__

以下は、LangChainを使用したLCELでのシステムプロンプトとユーザープロンプトを設定し、文字列を取り出す例です。LCELによって、| でチェインを繋ぐまでが、冗長になりがちです。またインポート場所も変わったりするので、追従が大変です。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser


# プロンプトテンプレートを作成
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("あなたは知識豊富なアシスタントです。簡潔かつ正確に回答してください。"),
    HumanMessagePromptTemplate.from_template("{question}")
])


# LLM（OpenAI GPT-4）を設定
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

# 出力パーサー
output_parser = StrOutputParser()

# LCEL を利用して、プロンプト → LLM → 出力パース の流れを構築
chain = prompt | llm | output_parser

# ユーザーの入力
user_input = {"question": "量子コンピュータとは何ですか？"}

# 実行
response = chain.invoke(user_input)
```

__SnappyChainの場合__

同じコードをSnappyChainで書きます。システムプロンプトをああして、ユーザープロンプトを、
SnappyChainを使うことで直感的に記載できます。また記憶負担を高めるように務めた関数名を採用します。

```python
from snappychain import system_prompt, human_prompt, openai_chat , output
chain = (
    system_prompt("あなたは知識豊富なアシスタントです。簡潔かつ正確に回答してください。") 
    | human_prompt("{question}") 
    | openai_chat(model="gpt-4", temperature=0.7) 
    | output()
)

# ユーザーの入力
user_input = {"question": "量子コンピュータとは何ですか？"}

# 実行
response = chain.invoke(user_input)
```

## インストール方法

### 前提条件

- Python 3.9以上
- pip（Pythonパッケージマネージャー）

### PyPIからのインストール（予定）

```bash
pip install snappychain
```

### 開発版のインストール

リポジトリからクローンしてインストールする場合：

```bash
git clone https://github.com/yourusername/snappychain.git
cd snappychain
pip install -e .
```

## クイックスタート

### 1. 基本的なLLMの利用

SnappyChainでは、OpenAIやOllamaなどのLLMを簡単に利用できます。基本的な使い方は以下の通りです：

```python
from snappychain.llm import get_llm

# OpenAI GPT-3.5-turboの設定
config = {
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "temperature": 0.7
}

# LLMインスタンスの取得
llm = get_llm(config)

# テキスト生成
response = llm.invoke("日本の四季について教えてください")
print(response)
```

## LCELによる簡単なLLM


```python
dev() \
    | sytem_prompt("あなたは優秀な") \
    | human_prompt("{question"})  \
    | openai_chat("gpt-4o")  \
    | output()
```



### 2. エンベディングの利用

テキストをベクトル化するためのエンベディングも簡単に利用できます：

```python
from snappychain.embedding import get_embeddings

# OpenAIのエンベディングを設定
config = {
    "provider": "openai",
    "model": "text-embedding-3-small"
}

# エンベディングインスタンスの取得
embeddings = get_embeddings(config)

# テキストのベクトル化
text = "これはサンプルテキストです"
vector = embeddings.embed_query(text)
```

## LCELを使った直感的なチェーン構築

LangChain Expression Language（LCEL）は、複雑なNLPタスクを直感的に構築するための強力な機能です。SnappyChainではLCELを活用して、シンプルで読みやすいコードでNLPパイプラインを構築することができます。

### 1. LCELの基本

LCELでは、`|`（パイプ）演算子を使用してコンポーネントを連結し、データの流れを直感的に表現できます：

```python
from snappychain.llm import get_llm
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# LLMの設定
llm_config = {
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "temperature": 0.7
}

# LLMインスタンスの取得
llm = get_llm(llm_config)

# プロンプトテンプレートの作成
prompt = ChatPromptTemplate.from_template("以下のトピックについて説明してください: {topic}")

# LCELチェーンの構築
chain = {"topic": RunnablePassthrough()} | prompt | llm

# チェーンの実行
response = chain.invoke("日本の歴史")
print(response)
```

### 2. dev()によるデバッグ

LCELの強力な機能の一つが`dev()`メソッドです。このメソッドを使用すると、チェーンの各ステップの入出力を視覚的に確認でき、デバッグが容易になります：

```python
# 開発モードでチェーンを実行
# ブラウザでインタラクティブなデバッグインターフェースが開きます
chain.dev(topic="人工知能の歴史")
```

### 3. VectorSearchの簡単化

LCELを使用すると、ベクトル検索を含むパイプラインも直感的に構築できます：

```python
from snappychain.embedding import get_embeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import os

# エンベディングの設定
embedding_config = {
    "provider": "openai",
    "model": "text-embedding-3-small"
}
embeddings = get_embeddings(embedding_config)

# サンプルドキュメント
documents = [
    Document(page_content="東京は日本の首都です"),
    Document(page_content="京都には多くの寺院があります"),
    Document(page_content="大阪はたこ焼きが有名です")
]

# ベクトルストアの作成
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

# ベクトル検索チェーン
search_chain = retriever | (lambda docs: "\n".join([doc.page_content for doc in docs]))

# 検索実行
results = search_chain.invoke("日本の都市")
print(results)
```

### 4. RAGの簡単実装

LCELを使うと、RAG（検索拡張生成）システムも非常に簡潔に実装できます：

```python
from snappychain.llm import get_llm
from langchain.prompts import ChatPromptTemplate

# LLMの設定
llm_config = {
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "temperature": 0.2
}
llm = get_llm(llm_config)

# プロンプトテンプレート
rag_prompt = ChatPromptTemplate.from_template("""
以下の情報を参照して質問に答えてください。

コンテキスト情報:
{context}

質問: {question}

回答:
""")

# RAGチェーンの構築
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": lambda x: x
    }
    | rag_prompt
    | llm
)

# RAGシステムの実行
response = rag_chain.invoke("日本の都市について教えてください")
print(response)
```

LCELを使用することで、従来は複雑だったNLPパイプラインの構築が直感的になり、コードの可読性と保守性が向上します。特にRAGシステムの構築において、その利点が顕著に現れます。

## ステップアップガイド：RAGシステムの構築

以下では、段階的にRAG（検索拡張生成）システムを構築する方法を解説します。

### ステップ1: 基本的なRAGの設定

まず、シンプルなRAGシステムを設定してみましょう：

```python
from snappychain.rag import build_rag_chain
from langchain.schema import Document

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
            "persist_dir": "./vectorstore"
        }
    }
}

# RAGインスタンスの構築
rag = build_rag_chain(config)

# ドキュメントの追加
documents = [
    Document(page_content="東京は日本の首都であり、世界で最も人口の多い都市の一つです。"),
    Document(page_content="京都は日本の古都であり、多くの寺院や神社があります。"),
    Document(page_content="大阪は日本の第二の都市であり、食文化で有名です。")
]
rag.add_documents(documents)

# クエリの実行
response = rag.query("日本の都市について教えてください")
print(response)
```

### ステップ2: 日本語対応のBM25SJリトリーバーの追加

日本語テキストに特化したBM25SJリトリーバーを追加してRAGの性能を向上させます：

```python
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
            "persist_dir": "./vectorstore"
        }
    },
    "retrievers": [
        {
            "provider": "BM25SJ",
            "settings": {
                "k1": 1.2,
                "b": 0.75,
                "k": 10,
                "save_dir": "./bm25store"
            }
        }
    ]
}

rag = build_rag_chain(config)
rag.add_documents(documents)

# 日本語クエリの実行
response = rag.query("京都の観光地を教えてください")
print(response)
```

### ステップ3: リランカーの追加

検索結果をより適切に並べ替えるためのリランカーを追加します：

```python
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
            "persist_dir": "./vectorstore"
        }
    },
    "retrievers": [
        {
            "provider": "BM25SJ",
            "settings": {
                "k1": 1.2,
                "b": 0.75,
                "k": 10,
                "save_dir": "./bm25store"
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

rag = build_rag_chain(config)
rag.add_documents(documents)

# クエリの実行
response = rag.query("日本の伝統文化について教えてください")
print(response)
```

### ステップ4: 並列クエリ処理

複数のクエリを効率的に処理するための並列処理機能を使用します：

```python
# 複数のクエリを並列に処理
queries = [
    "東京について教えてください",
    "京都の観光スポットは？",
    "大阪の食文化とは？"
]

# 最大3つのスレッドで並列処理
results = rag.batch_query(queries, max_workers=3)

# 結果の表示
for result in results:
    print(f"質問: {result['query']}")
    print(f"応答: {result['result']}")
    print(f"処理時間: {result['time']:.2f}秒")
    print("-" * 50)
```

## 応用例: 完全なRAGアプリケーション

以下は、すべての機能を組み合わせた完全なRAGアプリケーションの例です：

```python
import os
import time
from langchain.schema import Document
from snappychain.rag import build_rag_chain

# ディレクトリの作成
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

# サンプルドキュメントの作成
documents = [
    Document(
        page_content="東京は日本の首都であり、世界で最も人口の多い都市の一つです。東京スカイツリーや東京タワーなどの観光名所があります。",
        metadata={"source": "travel_guide", "language": "ja"}
    ),
    Document(
        page_content="京都は日本の古都であり、金閣寺や清水寺など多くの寺院や神社があります。春の桜と秋の紅葉が特に美しいことで知られています。",
        metadata={"source": "travel_guide", "language": "ja"}
    ),
    Document(
        page_content="大阪は日本の第二の都市であり、たこ焼きやお好み焼きなどの食文化で有名です。大阪城や道頓堀などの観光スポットがあります。",
        metadata={"source": "travel_guide", "language": "ja"}
    ),
    Document(
        page_content="富士山は日本の最高峰であり、標高3,776メートルです。世界文化遺産に登録されており、多くの観光客が訪れます。",
        metadata={"source": "nature_guide", "language": "ja"}
    ),
    Document(
        page_content="寿司は酢飯に魚介類や野菜を組み合わせた日本の伝統的な料理です。江戸時代から発展し、現在では世界中で人気があります。",
        metadata={"source": "food_guide", "language": "ja"}
    )
]

# RAGインスタンスの構築とドキュメントの追加
rag = build_rag_chain(config)
rag.add_documents(documents)

def interactive_qa():
    """対話式Q&Aセッション"""
    print("SnappyChain RAGデモ（終了するには 'exit' と入力）")
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

if __name__ == "__main__":
    interactive_qa()
```

## トラブルシューティング

### よくある問題と解決策

1. **APIキーが設定されていない**
   - 問題: `API key not found` などのエラーが表示される
   - 解決策: 環境変数に適切なAPIキーを設定してください
     ```bash
     export OPENAI_API_KEY="your-api-key-here"
     ```

2. **メモリ不足エラー**
   - 問題: 大量のドキュメントを処理する際にメモリ不足になる
   - 解決策: ドキュメントのバッチサイズを小さくするか、チャンク数を減らしてください

3. **ベクトルストアの保存エラー**
   - 問題: ベクトルストアの保存先ディレクトリがない
   - 解決策: 保存先ディレクトリが存在することを確認し、必要に応じて作成してください
     ```python
     import os
     os.makedirs("./vectorstore", exist_ok=True)
     ```

## まとめ

このガイドでは、SnappyChainライブラリの基本的な使い方から高度なRAGシステムの構築方法までを解説しました。SnappyChainを活用することで、日本語と英語の両方に対応した高性能なNLPアプリケーションを簡単に構築することができます。

詳細な使用方法やAPI仕様については、公式ドキュメントをご参照ください。

## 次のステップ

- [詳細なAPI仕様](../api/index.md)
- [高度な使用例](../examples/index.md)
- [カスタムリトリーバーの作成](../advanced/custom_retrievers.md)
- [パフォーマンスチューニングガイド](../advanced/performance.md) 