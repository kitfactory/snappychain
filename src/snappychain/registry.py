import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from .print import debug_print, Color

# コンポーネントレジストリ
# Component registry
class ComponentRegistry:
    """
    モデル、プロンプトテンプレート、およびチェイン関連オブジェクトを管理するレジストリ
    Registry to manage models, prompt templates, and chain-related objects
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ComponentRegistry, cls).__new__(cls)
            cls._instance.chain_objects = {}  # オブジェクトのキャッシュ / Object cache
            cls._instance.access_times = {}  # 最終アクセス時間を記録 / Record last access time
            cls._instance.verbose = False
            
            # 環境変数からLRUキャッシュの保持時間を取得（デフォルト24時間）
            # Get LRU cache retention time from environment variable (default 24 hours)
            try:
                cls._instance.lru_hours = int(os.environ.get('CHAIN_CACHE_LRU_HOUR', 24))
            except (ValueError, TypeError):
                cls._instance.lru_hours = 24
                debug_print("環境変数の解析エラー / Environment variable parsing error", 
                           f"CHAIN_CACHE_LRU_HOUR: デフォルト値の24時間を使用します / Using default value of 24 hours", 
                           Color.YELLOW)
        return cls._instance
    
    def set_verbose(self, verbose):
        """
        詳細出力モードを設定する
        Set verbose mode
        
        Args:
            verbose (bool): 詳細出力モードかどうか / Whether in verbose mode
        """
        self.verbose = verbose
    
    def generate_id(self, data):
        """
        データから短い識別子を生成する
        Generate a short identifier from data
        
        Args:
            data: 識別子を生成するためのデータ / Data to generate identifier from
            
        Returns:
            str: 生成された短い識別子 / Generated short identifier
        """
        # データをJSON文字列に変換
        # Convert data to JSON string
        if isinstance(data, dict) or isinstance(data, list):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        # ハッシュ値を計算
        # Calculate hash value
        hash_obj = hashlib.md5(data_str.encode())
        hash_hex = hash_obj.hexdigest()
        
        # 短い識別子（最初の16文字）を返す
        # Return short identifier (first 16 characters)
        return hash_hex[:16]
    
    def _clean_expired_objects(self):
        """
        期限切れのオブジェクトをキャッシュから削除する
        Remove expired objects from cache
        """
        now = datetime.now()
        expiration_delta = timedelta(hours=self.lru_hours)
        expired_keys = []
        
        # 期限切れのオブジェクトを特定
        # Identify expired objects
        for key, access_time in self.access_times.items():
            if now - access_time > expiration_delta:
                expired_keys.append(key)
        
        # 期限切れのオブジェクトを削除
        # Remove expired objects
        for key in expired_keys:
            if key in self.chain_objects:
                if self.verbose:
                    debug_print(f"期限切れのオブジェクトを削除 / Removing expired object", 
                               f"Key: {key}", Color.YELLOW)
                del self.chain_objects[key]
                del self.access_times[key]
    
    def get_object(self, key: Any) -> Optional[Any]:
        """
        キーに対応したオブジェクトを取得する
        Get object corresponding to key
        
        Args:
            key (Any): キー / Key
            
        Returns:
            Any: 対応するオブジェクト、存在しない場合はNone / Corresponding object, None if not exists
        """
        # 期限切れのオブジェクトをクリーンアップ
        # Clean up expired objects
        self._clean_expired_objects()
        
        if key in self.chain_objects:
            # アクセス時間を更新
            # Update access time
            self.access_times[key] = datetime.now()
            
            if self.verbose:
                debug_print(f"オブジェクトを取得 / Getting object", 
                           f"Key: {key}", Color.CYAN)
            
            return self.chain_objects[key]
        
        return None
    
    def set_object(self, key: Any, obj: Any) -> None:
        """
        キーに対応したオブジェクトを設定する
        Set object corresponding to key
        
        Args:
            key (Any): キー / Key
            obj (Any): 設定するオブジェクト / Object to set
        """
        # 期限切れのオブジェクトをクリーンアップ
        # Clean up expired objects
        self._clean_expired_objects()
        
        self.chain_objects[key] = obj
        self.access_times[key] = datetime.now()
        
        if self.verbose:
            debug_print(f"オブジェクトを設定 / Setting object", 
                       f"Key: {key}", Color.CYAN)
    
    def get_chain_object(self, chain_id: str, index: int) -> Optional[Any]:
        """
        チェインIDとインデックスに対応したオブジェクトを取得する
        Get object corresponding to chain ID and index
        
        Args:
            chain_id (str): チェインID / Chain ID
            index (int): インデックス / Index
            
        Returns:
            Any: 対応するオブジェクト、存在しない場合はNone / Corresponding object, None if not exists
        """
        key = (chain_id, index)
        return self.get_object(key)
    
    def set_chain_object(self, chain_id: str, index: int, obj: Any) -> None:
        """
        チェインIDとインデックスに対応したオブジェクトを設定する
        Set object corresponding to chain ID and index
        
        Args:
            chain_id (str): チェインID / Chain ID
            index (int): インデックス / Index
            obj (Any): 設定するオブジェクト / Object to set
        """
        key = (chain_id, index)
        self.set_object(key, obj)
    
    def get_chain_objects(self, chain_id: str) -> List[Tuple[int, Any]]:
        """
        指定されたチェインIDに関連するすべてのオブジェクトを取得する
        Get all objects related to the specified chain ID
        
        Args:
            chain_id (str): チェインID / Chain ID
            
        Returns:
            List[Tuple[int, Any]]: インデックスとオブジェクトのペアのリスト / List of pairs of index and object
        """
        # 期限切れのオブジェクトをクリーンアップ
        # Clean up expired objects
        self._clean_expired_objects()
        
        result = []
        for key in list(self.chain_objects.keys()):
            if isinstance(key, tuple) and len(key) == 2 and key[0] == chain_id:
                cid, idx = key
                # アクセス時間を更新
                # Update access time
                self.access_times[key] = datetime.now()
                result.append((idx, self.chain_objects[key]))
        
        # インデックス順にソート
        # Sort by index
        result.sort(key=lambda x: x[0])
        
        if self.verbose and result:
            debug_print(f"チェインに関連するオブジェクトを取得 / Getting objects related to chain", 
                       f"Chain ID: {chain_id}, Object count: {len(result)}", Color.CYAN)
        
        return result
    
    def remove_chain_objects(self, chain_id: str) -> int:
        """
        指定されたチェインIDに関連するすべてのオブジェクトを削除する
        Remove all objects related to the specified chain ID
        
        Args:
            chain_id (str): チェインID / Chain ID
            
        Returns:
            int: 削除されたオブジェクトの数 / Number of objects removed
        """
        keys_to_remove = []
        for key in list(self.chain_objects.keys()):
            if isinstance(key, tuple) and len(key) == 2 and key[0] == chain_id:
                keys_to_remove.append(key)
        
        # オブジェクトとアクセス時間を削除
        # Remove objects and access times
        for key in keys_to_remove:
            del self.chain_objects[key]
            del self.access_times[key]
        
        if self.verbose and keys_to_remove:
            debug_print(f"チェインに関連するオブジェクトを削除 / Removing objects related to chain", 
                       f"Chain ID: {chain_id}, Removed count: {len(keys_to_remove)}", Color.YELLOW)
        
        return len(keys_to_remove)
    
    def get_model(self, model_type, **kwargs):
        """
        モデルを取得または作成する
        Get or create a model
        
        Args:
            model_type (str): モデルの種類 (openai, anthropic, ollama など) / Model type (openai, anthropic, ollama, etc.)
            **kwargs: モデルのパラメータ / Model parameters
            
        Returns:
            モデルインスタンス / Model instance
        """
        # モデルパラメータから識別子を生成
        # Generate identifier from model parameters
        model_params = {"type": model_type, **kwargs}
        model_id = self.generate_id(model_params)
        
        # キャッシュからモデルを取得
        # Get model from cache
        model = self.get_object(("model", model_id))
        
        if model is None:
            if self.verbose:
                debug_print(f"モデルを新規作成 / Creating new model", f"{model_type} [ID: {model_id}]", Color.CYAN)
            
            # モデルの種類に応じてインスタンスを作成
            # Create instance based on model type
            if model_type == "openai":
                model = ChatOpenAI(model_name=kwargs.get("model_name", "gpt-4o-mini"), 
                                  temperature=kwargs.get("temperature", 0.7))
            elif model_type == "anthropic":
                model = ChatAnthropic(model_name=kwargs.get("model_name", "claude-3-haiku-20240307"), 
                                     temperature=kwargs.get("temperature", 0.7))
            elif model_type == "ollama":
                model = ChatOllama(model=kwargs.get("model_name", "llama3"), 
                                  temperature=kwargs.get("temperature", 0.7))
            else:
                raise ValueError(f"不明なモデルタイプ / Unknown model type: {model_type}")
            
            # モデルをキャッシュに保存
            # Save model to cache
            self.set_object(("model", model_id), model)
        elif self.verbose:
            debug_print(f"キャッシュからモデルを取得 / Getting model from cache", f"{model_type} [ID: {model_id}]", Color.CYAN)
        
        return model
    
    def get_prompt(self, messages):
        """
        プロンプトテンプレートを取得または作成する
        Get or create a prompt template
        
        Args:
            messages (list): メッセージのリスト / List of messages
            
        Returns:
            ChatPromptTemplate: プロンプトテンプレートインスタンス / Prompt template instance
        """
        # メッセージから識別子を生成
        # Generate identifier from messages
        prompt_id = self.generate_id(messages)
        
        # キャッシュからプロンプトテンプレートを取得
        # Get prompt template from cache
        prompt = self.get_object(("prompt", prompt_id))
        
        if prompt is None:
            if self.verbose:
                debug_print(f"プロンプトテンプレートを新規作成 / Creating new prompt template", f"[ID: {prompt_id}]", Color.CYAN)
            prompt = ChatPromptTemplate.from_messages(messages)
            
            # プロンプトテンプレートをキャッシュに保存
            # Save prompt template to cache
            self.set_object(("prompt", prompt_id), prompt)
        elif self.verbose:
            debug_print(f"キャッシュからプロンプトテンプレートを取得 / Getting prompt template from cache", f"[ID: {prompt_id}]", Color.CYAN)
        
        return prompt

# レジストリのインスタンスを作成
# Create registry instance
registry = ComponentRegistry()
