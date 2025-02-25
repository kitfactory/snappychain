"""
Unified document reranking implementation.
統一されたドキュメント再ランキングの実装。
"""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from langchain_core.documents.base import Document
from langchain_cohere import CohereRerank
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from onelogger import Logger
from snappychain.prompt import system_prompt, human_prompt
from snappychain.schema import schema
from snappychain.chat import openai_chat
from snappychain.output import output

logger = Logger.get_logger(__name__)

"""
Configuration example:
{
    "provider": "cohere"|"llm",
    "settings": {
        "model": "model_name", // Required for both providers
        "top_n": 5, // Optional, defaults to 5
        "threshold": 0.7 // Optional, defaults to null for cohere, not used for llm
    }
}
"""

@dataclass
class UnifiedRerank:
    """
    A unified document reranker that supports both Cohere and LLM-based reranking.
    Cohereおよび LLM ベースの再ランキングの両方をサポートする統一されたドキュメント再ランカー。

    Attributes:
        config (Dict): Configuration for the reranker
                      再ランカーの設定
        compressor (Any): The underlying document compressor
                        基礎となるドキュメントコンプレッサー
    """
    config: Dict[str, Any]
    compressor: Any = None
    rerank_chain: Any = None
    _top_n: int = 5

    def __post_init__(self):
        """
        Initialize the reranker based on the configuration.
        設定に基づいて再ランカーを初期化します。
        """
        try:
            provider = self.config.get("provider")
            settings = self.config.get("settings", {})
            
            if provider == "cohere":
                model = settings.get("model")
                top_n = settings.get("top_n", 5)
                threshold = settings.get("threshold")
                
                if not model:
                    raise ValueError("Model name is required for Cohere reranker / Cohereリランカーにはモデル名が必要です")
                
                self.compressor = CohereRerank(
                    model=model,
                    top_n=top_n,
                    relevance_threshold=threshold
                )
                
            elif provider == "llm":
                model = settings.get("model")
                llm_provider = settings.get("llm_provider", "openai")
                top_n = settings.get("top_n", 5)
                
                if not model:
                    raise ValueError("Model name is required for LLM reranker / LLMリランカーにはモデル名が必要です")
                
                # スキーマの定義
                # Define schema for relevance ranking
                schema_list = [
                    {
                        "name": "rankings",
                        "description": "An array of document indices ranked by relevance (0-based, most relevant first)",
                        "type": "list[integer]"
                    },
                    {
                        "name": "explanations",
                        "description": "Brief explanations for why each document was ranked in this position",
                        "type": "list[string]"
                    }
                ]
                
                # LLMリランク用のチェーンを構築
                # Build chain for LLM reranking
                self.rerank_chain = (
                    schema(schema_list)
                    | system_prompt("You are a document relevance ranking assistant. Your task is to rank documents based on their relevance to a given query. Return the indices of documents in order of relevance (most relevant first).")
                    | openai_chat(model=model)
                    | output("json")
                )
                
                # Store the top_n value to be used in compress_documents
                self._top_n = top_n
                
            else:
                raise ValueError(f"Unsupported reranker provider: {provider} / サポートされていない再ランカープロバイダー: {provider}")
                
        except Exception as e:
            logger.error("\033[31mError initializing reranker: %s\033[0m", str(e))
            raise
    
    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        """
        Compress (rerank) the documents based on their relevance to the query.
        クエリに対する関連性に基づいてドキュメントを圧縮（再ランク）します。

        Args:
            documents (List[Document]): List of documents to rerank
                                      再ランクするドキュメントのリスト
            query (str): The query to use for reranking
                       再ランクに使用するクエリ

        Returns:
            List[Document]: The reranked documents
                          再ランクされたドキュメント
        """
        try:
            if not documents:
                logger.warning("No documents provided for reranking / 再ランク用のドキュメントが提供されていません")
                return []
            
            provider = self.config.get("provider")
            
            if provider == "cohere":
                if not self.compressor:
                    raise ValueError("Compressor not initialized / コンプレッサーが初期化されていません")
                return self.compressor.compress_documents(documents, query)
                
            elif provider == "llm" and self.rerank_chain:
                # LLMチェーンを使用して再ランキング処理を実行
                # Use LLM chain to perform reranking
                
                # ドキュメントの内容をプロンプトに追加
                # Add document contents to the prompt
                docs_text = []
                for i, doc in enumerate(documents):
                    # ドキュメントのページコンテンツまたはコンテンツを取得
                    # Get page_content or content from the document
                    content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                    docs_text.append(f"[Document {i}]: {content}")
                
                docs_prompt = "\n\n".join(docs_text)
                
                # リランクチェーンを実行
                # Execute rerank chain
                prompt = f"Query: {query}\n\nDocuments to rank:\n{docs_prompt}\n\nRank these documents based on their relevance to the query."
                
                result = self.rerank_chain.invoke({
                    "_session": {
                        "prompt": [
                            {
                                "human": prompt
                            }
                        ]
                    },
                    "_dev": self.config.get("debug", False)
                })
                
                # 結果からランキングを取得
                # Get rankings from result
                if not isinstance(result, dict) or "rankings" not in result:
                    logger.warning("Invalid result from LLM reranker, returning original documents / LLMリランカーから無効な結果、元のドキュメントを返します")
                    return documents[:self._top_n]
                
                # ランキングに基づいてドキュメントを並べ替え
                # Reorder documents based on rankings
                rankings = result["rankings"]
                
                # インデックスの検証（範囲外のインデックスを除外）
                # Validate indices (exclude out-of-range indices)
                valid_rankings = [idx for idx in rankings if 0 <= idx < len(documents)]
                
                # 上位N件のドキュメントを取得
                # Get top N documents
                reranked_docs = [documents[idx] for idx in valid_rankings[:self._top_n]]
                
                # ランキングに含まれていない残りのドキュメントを追加（もし結果が少なすぎる場合）
                # Add remaining documents not included in rankings (if results are too few)
                if len(reranked_docs) < self._top_n:
                    remaining_indices = [i for i in range(len(documents)) if i not in valid_rankings]
                    remaining_docs = [documents[idx] for idx in remaining_indices]
                    reranked_docs.extend(remaining_docs[:self._top_n - len(reranked_docs)])
                
                # 説明をメタデータに追加（存在する場合）
                # Add explanations to metadata if available
                if "explanations" in result and isinstance(result["explanations"], list):
                    explanations = result["explanations"]
                    for i, doc in enumerate(reranked_docs):
                        if i < len(explanations):
                            if not hasattr(doc, "metadata"):
                                doc.metadata = {}
                            doc.metadata["rerank_explanation"] = explanations[i]
                
                return reranked_docs
                
            else:
                logger.warning("No valid reranker configured, returning original documents / 有効なリランカーが設定されていません。元のドキュメントを返します")
                return documents[:min(self._top_n, len(documents))]
                
        except Exception as e:
            logger.error("\033[31mError during document reranking: %s\033[0m", str(e))
            # エラーが発生した場合は元のドキュメントを返す
            # Return original documents in case of error
            return documents[:min(self._top_n if hasattr(self, "_top_n") else 5, len(documents))]

def build_reranker(config: Dict[str, Any]) -> UnifiedRerank:
    """
    Build a unified reranker from configuration.
    設定から統一された再ランカーを構築します。

    Args:
        config (Dict[str, Any]): Configuration dictionary
                                設定辞書

    Returns:
        UnifiedRerank: Initialized reranker
                      初期化された再ランカー
    """
    try:
        # Validate config
        # 設定の検証
        required_keys = ["provider", "settings"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key} / 必要な設定キーがありません: {key}")
                
        # Validate provider
        # プロバイダーの検証
        if config["provider"] not in ["cohere", "llm"]:
            raise ValueError("Provider must be 'cohere' or 'llm' / プロバイダーは 'cohere' または 'llm' である必要があります")
            
        # Validate settings
        # 設定の検証
        settings = config.get("settings", {})
        if "model" not in settings:
            raise ValueError("Model name is required in settings / 設定にモデル名が必要です")
            
        # Create reranker
        # 再ランカーの作成
        return UnifiedRerank(config=config)
        
    except Exception as e:
        logger.error("\033[31mError building reranker: %s\033[0m", str(e))
        raise 