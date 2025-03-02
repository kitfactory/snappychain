# SnappyChain API概要

このドキュメントでは、SnappyChainライブラリの主要なAPIコンポーネントについて概要を説明します。より詳細なAPI仕様については、各モジュールのドキュメントを参照してください。

## 主要コンポーネント

SnappyChainライブラリは、以下の主要コンポーネントで構成されています：

| コンポーネント | 説明 |
|--------------|------|
| `rag` | RAG（検索拡張生成）システムの中核機能を提供するモジュール |
| `bm25sj` | 日本語に最適化されたBM25検索リトリーバー |
| `llm` | 言語モデル（LLM）とのインターフェース |
| `embeddings` | テキストをベクトル化するためのエンベディング機能 |
| `utils` | ユーティリティ関数のコレクション |

## RAGモジュール

RAGモジュールは、検索拡張生成システムを構築するための中核機能を提供します。

### 主要クラスとメソッド

| クラス/関数 | 説明 |
|------------|------|
| `Rag` | 複数のリトリーバーとLLMを組み合わせたモジュラーRAGシステムを実装するクラス |
| `build_rag_chain` | 設定から完全なRAGシステムを構築するヘルパー関数 |

#### Ragクラス

```python
class Rag:
    def __init__(self, config: Dict[str, Any])
    def add_documents(self, documents: List[Document]) -> 'Rag'
    def query(self, question: str, top_k: int = 4) -> str
    def batch_query(self, questions: List[str], max_workers: int = 3) -> List[Dict[str, Any]]
```

#### build_rag_chain関数

```python
def build_rag_chain(config: Dict[str, Any]) -> Rag
```

## BM25SJリトリーバー

BM25SJリトリーバーは、日本語テキストに最適化されたBM25アルゴリズムを使用した検索リトリーバーです。

### 主要クラスとメソッド

```python
class BM25SJRetriever(BaseRetriever):
    def __init__(self, k1: float = 1.2, b: float = 0.75, k: int = 10, save_dir: Optional[str] = None)
    def add_documents(self, documents: List[Document]) -> None
    def get_relevant_documents(self, query: str) -> List[Document]
```

## 設定形式

SnappyChainでは、JSONスタイルの設定オブジェクトを使用して、さまざまなコンポーネントを設定します。以下に一般的な設定形式を示します：

```python
config = {
    "llm": {
        "provider": "openai"|"ollama",
        "model": "model/name",
        "temperature": 0.2
    },
    "embeddings": {
        "provider": "openai"|"ollama",
        "model": "model/name"
    },
    "vector_store": {
        "provider": "FAISS"|"Chroma",
        "settings": {
            "persist_dir": "path/to/persist/dir"
        }
    },
    "retrievers": [
        {
            "provider": "BM25SJ",
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
        "llm_provider": "openai",
        "model": "model/name",
        "temperature": 0.0
    }
}
```

## クラス図

以下は、SnappyChainの主要コンポーネント間の関係を示す簡略化されたクラス図です：

```plantuml
@startuml

package "snappychain" {
    class Rag {
        + embeddings: Embeddings
        + vector_store: VectorStore
        + llm: BaseLanguageModel
        + retrievers: List[BaseRetriever]
        + reranker: Optional[Callable]
        + write_lock: threading.Lock
        + __init__(config: Dict[str, Any])
        + add_documents(documents: List[Document]): Rag
        + query(question: str, top_k: int): str
        + batch_query(questions: List[str], max_workers: int): List[Dict[str, Any]]
    }
    
    class BM25SJRetriever {
        + k1: float
        + b: float
        + k: int
        + save_dir: Optional[str]
        + __init__(k1: float, b: float, k: int, save_dir: Optional[str])
        + add_documents(documents: List[Document]): None
        + get_relevant_documents(query: str): List[Document]
    }
    
    interface BaseRetriever {
        + get_relevant_documents(query: str): List[Document]
    }
    
    function build_rag_chain(config: Dict[str, Any]): Rag
    
    BM25SJRetriever --|> BaseRetriever
    Rag o-- BaseRetriever
    build_rag_chain --> Rag
}

@enduml
```

## 次のステップ

APIの詳細については、以下のドキュメントを参照してください：

- [RAGモジュール詳細](../api/rag.md)
- [BM25SJリトリーバー詳細](../api/bm25sj.md)
- [設定リファレンス](../api/config.md)
- [高度な使用例](../examples/index.md) 