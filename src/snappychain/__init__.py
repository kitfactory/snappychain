"""
snappychainパッケージ
snappychain package
"""

# 基本的なインポート
# Basic imports
import functools
import importlib
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic
from functools import wraps
import hashlib
import time
import inspect
from langchain.globals import set_debug

# チェイン関連の機能をchain.pyからインポート
# Import chain related functionality from chain.py
from .chain import (
    generate_chain_id,
    set_chain_id,
    get_chain_id,
    create_runnable,
    Chain,
    chain
)

# 通常のインポートを使用して他のモジュールをインポート
# Import other modules using regular import
from .chat import openai_chat
from .prompt import system_prompt, human_prompt, ai_prompt
from .schema import schema
from .devmode import validate
from .output import output
from .embedding import add_documents_to_vector_store, persist_vector_store, query_vector_store, openai_embedding, ollama_embedding
from .epistemize import epistemize
from .print import is_verbose, debug_print, debug_request, debug_response, debug_error, debug_info, Color

# ローダー関連のインポート
# Import for loader related functions
from .loader import text_load, pypdf_load, markitdown_load, get_chain_documents, directory_load, unstructured_markdown_load

# スプリッター関連のインポート
# Import for splitter related functions
from .splitter import split_text, recursive_split_text, markdown_text_splitter, python_text_splitter, json_text_splitter

# その他のインポート
# Import for other functions
from .wikipedia import wikipedia_to_text
from .rag import build_rag_chain
from .vectorstore import UnifiedVectorStore
from .rerank import UnifiedRerank
from .bm25sj import BM25SJRetriever, bm25sj, bm25sj_query


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
