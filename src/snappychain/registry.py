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
    
    def get_chain_object(self, chain_id: str, index: int, key: str = "default") -> Optional[Any]:
        """
        チェインIDとインデックスとキーに対応したオブジェクトを取得する
        Get object corresponding to chain ID, index and key
        
        Args:
            chain_id (str): チェインID / Chain ID
            index (int): インデックス / Index
            key (str): キー（デフォルトは"default"） / Key (default is "default")
            
        Returns:
            Any: 対応するオブジェクト、存在しない場合はNone / Corresponding object, None if not exists
        """
        composite_key = (chain_id, index, key)
        return self.get_object(composite_key)
    
    def set_chain_object(self, chain_id: str, index: int, obj: Any, key: str = "default") -> None:
        """
        チェインIDとインデックスとキーに対応したオブジェクトを設定する
        Set object corresponding to chain ID, index and key
        
        Args:
            chain_id (str): チェインID / Chain ID
            index (int): インデックス / Index
            obj (Any): 設定するオブジェクト / Object to set
            key (str): キー（デフォルトは"default"） / Key (default is "default")
        """
        composite_key = (chain_id, index, key)
        self.set_object(composite_key, obj)
    
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
            if isinstance(key, tuple) and len(key) == 3 and key[0] == chain_id:
                cid, idx, _ = key
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
            if isinstance(key, tuple) and len(key) == 3 and key[0] == chain_id:
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

# レジストリのインスタンスを作成
# Create registry instance
registry = ComponentRegistry()
