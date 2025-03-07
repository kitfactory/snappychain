"""
snappychainパッケージ
snappychain package
"""

# チェインID管理機能とチェイン基本機能
# Chain ID management and basic chain functionality
from langchain_core.runnables import RunnableLambda
import uuid
import functools
import importlib
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic
from functools import wraps
import hashlib
import time
import inspect
from langchain.globals import set_debug

# チェインIDを管理するグローバル辞書
# Global dictionary to manage chain IDs
_chain_registry = {}


def generate_chain_id():
    """
    新しいチェインIDを生成する
    Generate a new chain ID
    
    Returns:
        str: チェインID / Chain ID
    """
    return str(uuid.uuid4())[:8]


def set_chain_id(chain, chain_id=None):
    """
    チェインにIDを設定する
    Set ID on a chain
    
    Args:
        chain: チェイン / Chain
        chain_id (str, optional): チェインID / Chain ID
    
    Returns:
        str: チェインID / Chain ID
    """
    chain_id = chain_id or generate_chain_id()
    _chain_registry[id(chain)] = chain_id
    return chain_id


def get_chain_id(chain):
    """
    チェインからIDを取得する
    Get ID from a chain
    
    Args:
        chain: チェイン / Chain
    
    Returns:
        str: チェインID / Chain ID
    """
    return _chain_registry.get(id(chain), "unknown")


def create_runnable(func, chain_id=None):
    """
    関数をRunnableLambdaでラップして返す
    Wrap a function with RunnableLambda and return it
    
    Args:
        func: ラップする関数 / Function to wrap
        chain_id (str, optional): チェインID / Chain ID
    
    Returns:
        RunnableLambda: ラップされた関数 / Wrapped function
    """
    runnable = RunnableLambda(func)
    set_chain_id(runnable, chain_id)
    return runnable


def _override_pipe_operator():
    """
    パイプ演算子をオーバーライドする関数
    Function to override pipe operator
    """
    original_pipe = RunnableLambda.__or__
    def pipe_override(self, other):
        """
        パイプ演算子（|）をオーバーロードして、チェインIDを伝播する
        Overload pipe operator (|) to propagate chain ID
        
        Args:
            self: 左側のオペランド / Left operand
            other: 右側のオペランド / Right operand
        
        Returns:
            チェイン / Chain
        """
        chain = original_pipe(self, other)
        chain_id = get_chain_id(self)
        if chain_id != "unknown":
            set_chain_id(chain, chain_id)
        return chain
    RunnableLambda.__or__ = pipe_override


# パイプ演算子をオーバーライド
# Override pipe operator
_override_pipe_operator()


class Chain(RunnableLambda):
    """
    チェインクラス
    Chain class
    """
    def __init__(self, func, chain_id=None):
        """
        初期化
        Initialization
        
        Args:
            func: ラップする関数 / Function to wrap
            chain_id (str, optional): チェインID / Chain ID
        """
        super().__init__(func)
        self.chain_id = set_chain_id(self, chain_id)

    def __or__(self, other):
        """
        パイプ演算子（|）をオーバーロード
        Overload pipe operator (|)
        
        Args:
            other: 右側のオペランド / Right operand
        
        Returns:
            Chain: 新しいチェイン / New chain
        """
        chain = super().__or__(other)
        chain.chain_id = self.chain_id
        return chain

    def __repr__(self):
        """
        文字列表現
        String representation
        
        Returns:
            str: 文字列表現 / String representation
        """
        return f"Chain(chain_id={self.chain_id}, func={self.func.__name__})"


def chain(func, chain_id=None):
    """
    関数をChainでラップして返す
    Wrap a function with Chain and return it
    
    Args:
        func: ラップする関数 / Function to wrap
        chain_id (str, optional): チェインID / Chain ID
    
    Returns:
        Chain: ラップされた関数 / Wrapped function
    """
    return Chain(func, chain_id)


# 遅延インポート関数
# Lazy import function
def lazy_import(module_name: str, attribute_name: Optional[str] = None):
    """
    モジュールを遅延インポートする
    Lazy import a module
    
    Args:
        module_name (str): モジュール名 / Module name
        attribute_name (Optional[str], optional): 属性名 / Attribute name
        
    Returns:
        Any: インポートされたモジュールまたは属性 / Imported module or attribute
    """
    def _import():
        module = importlib.import_module(module_name)
        if attribute_name:
            return getattr(module, attribute_name)
        return module
    
    return _import


# 遅延インポートを使用して他のモジュールをインポート
# Import other modules using lazy import
openai_chat = lazy_import(".chat", "openai_chat")
system_prompt = lazy_import(".prompt", "system_prompt")
human_prompt = lazy_import(".prompt", "human_prompt")
ai_prompt = lazy_import(".prompt", "ai_prompt")
schema = lazy_import(".schema", "schema")
validate = lazy_import(".devmode", "validate")
output = lazy_import(".output", "output")
add_documents_to_vector_store = lazy_import(".embedding", "add_documents_to_vector_store")
persist_vector_store = lazy_import(".embedding", "persist_vector_store")
query_vector_store = lazy_import(".embedding", "query_vector_store")
openai_embedding = lazy_import(".embedding", "openai_embedding")
ollama_embedding = lazy_import(".embedding", "ollama_embedding")
epistemize = lazy_import(".epistemize", "epistemize")
is_verbose = lazy_import(".print", "is_verbose")
debug_print = lazy_import(".print", "debug_print")
debug_request = lazy_import(".print", "debug_request")
debug_response = lazy_import(".print", "debug_response")
debug_error = lazy_import(".print", "debug_error")
debug_info = lazy_import(".print", "debug_info")
Color = lazy_import(".print", "Color")

# ローダー関連の遅延インポート
# Lazy import for loader related functions
text_load = lazy_import(".loader", "text_load")
pypdf_load = lazy_import(".loader", "pypdf_load")
markitdown_load = lazy_import(".loader", "markitdown_load")
get_chain_documents = lazy_import(".loader", "get_chain_documents")
directory_load = lazy_import(".loader", "directory_load")
unstructured_markdown_load = lazy_import(".loader", "unstructured_markdown_load")

# スプリッター関連の遅延インポート
# Lazy import for splitter related functions
split_text = lazy_import(".splitter", "split_text")
recursive_split_text = lazy_import(".splitter", "recursive_split_text")
markdown_text_splitter = lazy_import(".splitter", "markdown_text_splitter")
python_text_splitter = lazy_import(".splitter", "python_text_splitter")
json_text_splitter = lazy_import(".splitter", "json_text_splitter")

# その他の遅延インポート
# Lazy import for other functions
wikipedia_to_text = lazy_import(".wikipedia", "wikipedia_to_text")
build_rag_chain = lazy_import(".rag", "build_rag_chain")
UnifiedVectorStore = lazy_import(".vectorstore", "UnifiedVectorStore")
UnifiedRerank = lazy_import(".rerank", "UnifiedRerank")
BM25SJRetriever = lazy_import(".bm25sj", "BM25SJRetriever")
bm25sj = lazy_import(".bm25sj", "bm25sj")
bm25sj_query = lazy_import(".bm25sj", "bm25sj_query")


# OPTIONSの定義
# Definition of OPTIONS
OPTIONS ={
    'openai': [
        'openai_chat',
        'openai_embedding'
    ],
    # 'ollama': [
    #     'ollama_chat',
    #     'ollama_embedding'
    # ],
    # 'gemini': [
    #     'gemini_chat',
    #     'gemini_embedding'
    # ],
    # 'anthropic': [
    #     'anthropic_chat',
    #     'anthropic_embedding'
    # ],
    'faiss': [
        'faiss_vs_store',
        'faiss_vs_query',
    ],
    'chroma': [
        'chroma_vs_store',
        'chroma_vs_query',
    ],
    'bm25sj': [
        'bm25sj_store',
        'bm25sj_query',
    ],
    'pypdf': [
        'pypdf_load',
    ],
    'markitdown': [
        'markitdown_load',
    ]
}


def check_option():
    """
    オプションをチェックする
    Check options
    """
    pass


# パッケージの公開インターフェース
# Public interface of the package
__all__ = [
    # チャット関連 / Chat related
    "openai_chat",
    # "ollama_chat",
    # "gemini_chat",
    # "anthropic_chat",
    
    # プロンプト関連 / Prompt related
    "system_prompt",
    "human_prompt",
    "ai_prompt",
    
    # スキーマ関連 / Schema related
    "schema",
    
    # 開発モード関連 / Development mode related
    "validate",
    
    # 出力関連 / Output related
    "output",
    
    # 埋め込み関連 / Embedding related
    "openai_embedding",
    "ollama_embedding",
    "add_documents_to_vector_store",
    "persist_vector_store",
    "query_vector_store",
    
    # エピステマイズ関連 / Epistemize related
    "epistemize",
    
    # ベクトルストア関連 / Vector store related
    "UnifiedVectorStore",
    
    # リランク関連 / Rerank related
    "UnifiedRerank",
    
    # BM25SJ関連 / BM25SJ related
    "BM25SJRetriever",
    "bm25sj",
    "bm25sj_query",
    
    # ローダー関連 / Loader related
    "text_load",
    "pypdf_load",
    "markitdown_load",
    "get_chain_documents",
    "directory_load",
    "unstructured_markdown_load",
    
    # スプリッター関連 / Splitter related
    "split_text",
    "recursive_split_text",
    "markdown_text_splitter",
    "python_text_splitter",
    "json_text_splitter",
    
    # Wikipedia関連 / Wikipedia related
    "wikipedia_to_text",
    
    # RAG関連 / RAG related
    "build_rag_chain",
    
    # デバッグ関連 / Debug related
    "set_debug",
    "is_verbose",
    "debug_print",
    "debug_request",
    "debug_response",
    "debug_error",
    "debug_info",
    "Color",
    
    # チェインID関連 / Chain ID related
    "generate_chain_id",
    "set_chain_id",
    "get_chain_id",
    "create_runnable",
    "Chain",
    "chain",
]
