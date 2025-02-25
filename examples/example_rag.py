#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAGシステムの使用例を示すスクリプト。
OpenAI LLM、OpenAI Embeddings、FAISS Vector Store、BM25SJリトリーバー、およびLLMベースのリランカーを使用します。
"""

import os
import time
import shutil
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from onelogger import Logger

# snappychainからインポート
from snappychain.rag import Rag, build_rag_chain
from snappychain.bm25sj import BM25SJRetriever

# ロガーのセットアップ
logger = Logger.get_logger(__name__)

# RAG設定
RAG_CONFIG = {
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
            "persist_dir": "./examples/vectorstores/faiss_store"
        }
    },
    "retrievers": [
        {
            "provider": "BM25SJ",
            "settings": {
                "k1": 1.2,
                "b": 0.75,
                "k": 10,
                "save_dir": "./examples/vectorstores/bm25_store"
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

def cleanup_old_data():
    """
    以前のデモのデータをクリーンアップします。
    """
    # FAISSベクトルストアのクリーンアップ
    faiss_dir = RAG_CONFIG["vector_store"]["settings"]["persist_dir"]
    if os.path.exists(faiss_dir):
        logger.info(f"古いFAISSストアの削除: {faiss_dir}")
        shutil.rmtree(faiss_dir)
    
    # BM25SJのクリーンアップ
    bm25_dir = RAG_CONFIG["retrievers"][0]["settings"]["save_dir"]
    if os.path.exists(bm25_dir):
        logger.info(f"古いBM25SJストアの削除: {bm25_dir}")
        shutil.rmtree(bm25_dir)
    
    # ディレクトリの作成
    os.makedirs(faiss_dir, exist_ok=True)
    os.makedirs(bm25_dir, exist_ok=True)

def load_sample_documents() -> List[Document]:
    """
    サンプルドキュメントをロードします。
    日本語と英語のドキュメントを含みます。

    Returns:
        List[Document]: ドキュメントのリスト
    """
    # 英語のサンプルドキュメント
    english_docs = [
        Document(page_content="Tokyo is the capital city of Japan and one of the most populous cities in the world.", 
                 metadata={"source": "travel_guide", "language": "en", "id": "doc1"}),
        Document(page_content="Mount Fuji is the highest mountain in Japan, standing at 3,776 meters.",
                 metadata={"source": "travel_guide", "language": "en", "id": "doc2"}),
        Document(page_content="Kyoto was the former capital of Japan and is famous for its numerous temples and shrines.",
                 metadata={"source": "travel_guide", "language": "en", "id": "doc3"}),
        Document(page_content="Sushi is a traditional Japanese dish consisting of vinegared rice combined with seafood or vegetables.",
                 metadata={"source": "food_guide", "language": "en", "id": "doc4"}),
        Document(page_content="The bullet train (Shinkansen) is a high-speed railway network in Japan that connects major cities.",
                 metadata={"source": "transport_guide", "language": "en", "id": "doc5"})
    ]
    
    # 日本語のサンプルドキュメント
    japanese_docs = [
        Document(page_content="東京は日本の首都で、世界で最も人口の多い都市の一つです。",
                 metadata={"source": "travel_guide", "language": "ja", "id": "doc6"}),
        Document(page_content="富士山は日本で最も高い山で、標高3,776メートルです。",
                 metadata={"source": "travel_guide", "language": "ja", "id": "doc7"}),
        Document(page_content="京都は日本の旧都であり、多くの寺院や神社で有名です。",
                 metadata={"source": "travel_guide", "language": "ja", "id": "doc8"}),
        Document(page_content="寿司は酢飯に魚介類や野菜を組み合わせた日本の伝統的な料理です。",
                 metadata={"source": "food_guide", "language": "ja", "id": "doc9"}),
        Document(page_content="新幹線は日本の主要都市を結ぶ高速鉄道網です。",
                 metadata={"source": "transport_guide", "language": "ja", "id": "doc10"})
    ]
    
    # 両方のドキュメントを結合
    return english_docs + japanese_docs

def demonstrate_rag():
    """
    RAGの機能をデモンストレーションします。
    """
    try:
        # 古いデータのクリーンアップ
        cleanup_old_data()
        
        logger.info("RAGデモを開始します")
        
        # RAGチェーンのビルド
        rag = build_rag_chain(RAG_CONFIG)
        
        # サンプルドキュメントのロード
        documents = load_sample_documents()
        logger.info(f"{len(documents)}個のサンプルドキュメントを読み込みました")
        
        # ドキュメントの追加
        logger.info("RAGにドキュメントを追加します")
        rag.store_documents(documents)
        
        # 英語クエリの実行
        english_query = "What is Mount Fuji?"
        logger.info(f"英語クエリ実行: '{english_query}'")
        start_time = time.time()
        english_response = rag.query(english_query)
        query_time = time.time() - start_time
        logger.info(f"応答（実行時間: {query_time:.2f}秒）: {english_response}")
        
        # 日本語クエリの実行
        japanese_query = "京都について教えてください"
        logger.info(f"日本語クエリ実行: '{japanese_query}'")
        start_time = time.time()
        japanese_response = rag.query(japanese_query)
        query_time = time.time() - start_time
        logger.info(f"応答（実行時間: {query_time:.2f}秒）: {japanese_response}")
        
        # 並列クエリのデモンストレーション
        logger.info("並列クエリの実行")
        queries = [
            "Tell me about Tokyo",
            "What is sushi?",
            "富士山について教えてください",
            "新幹線とは何ですか？",
            "日本の伝統的な食べ物について教えてください"
        ]
        
        start_time = time.time()
        
        def execute_query(query):
            try:
                query_start = time.time()
                result = rag.query(query)
                query_time = time.time() - query_start
                return {"query": query, "result": result, "time": query_time}
            except Exception as e:
                logger.error(f"クエリ '{query}' の実行中にエラーが発生しました: {str(e)}")
                return {"query": query, "result": f"エラー: {str(e)}", "time": 0}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(execute_query, queries))
        
        total_time = time.time() - start_time
        logger.info(f"並列クエリの合計実行時間: {total_time:.2f}秒")
        
        for result in results:
            logger.info(f"クエリ: '{result['query']}', 実行時間: {result['time']:.2f}秒")
            logger.info(f"応答: {result['result']}")
        
        # 新しいドキュメントの追加と再クエリ
        logger.info("新しいドキュメントをRAGに追加します")
        new_docs = [
            Document(page_content="大阪は日本の第二の都市であり、おいしい食べ物で有名です。",
                     metadata={"source": "travel_guide", "language": "ja", "id": "doc11"}),
            Document(page_content="Osaka is the second largest city in Japan and is famous for its delicious food.",
                     metadata={"source": "travel_guide", "language": "en", "id": "doc12"})
        ]
        rag.store_documents(new_docs)
        
        # 新しいドキュメントに関するクエリ
        osaka_query = "大阪について教えてください"
        logger.info(f"新しいドキュメントに関するクエリ: '{osaka_query}'")
        osaka_response = rag.query(osaka_query)
        logger.info(f"応答: {osaka_response}")
        
        logger.info("RAGデモを完了しました")
        
    except Exception as e:
        logger.error(f"RAGデモの実行中にエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    demonstrate_rag() 