import hashlib
import json
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from .verbose import debug_print, Color

# コンポーネントレジストリ
# Component registry
class ComponentRegistry:
    """
    モデルとプロンプトテンプレートを管理するレジストリ
    Registry to manage models and prompt templates
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ComponentRegistry, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance.prompts = {}
            cls._instance.verbose = False
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
        
        # 短い識別子（最初の8文字）を返す
        # Return short identifier (first 8 characters)
        return hash_hex[:8]
    
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
        
        if model_id not in self.models:
            if self.verbose:
                debug_print(f"モデルを新規作成 / Creating new model", f"{model_type} [ID: {model_id}]", Color.CYAN)
            
            # モデルの種類に応じてインスタンスを作成
            # Create instance based on model type
            if model_type == "openai":
                self.models[model_id] = ChatOpenAI(model_name=kwargs.get("model_name", "gpt-4o-mini"), 
                                                  temperature=kwargs.get("temperature", 0.7))
            elif model_type == "anthropic":
                self.models[model_id] = ChatAnthropic(model_name=kwargs.get("model_name", "claude-3-haiku-20240307"), 
                                                     temperature=kwargs.get("temperature", 0.7))
            elif model_type == "ollama":
                self.models[model_id] = ChatOllama(model=kwargs.get("model_name", "llama3"), 
                                                  temperature=kwargs.get("temperature", 0.7))
            else:
                raise ValueError(f"不明なモデルタイプ / Unknown model type: {model_type}")
        elif self.verbose:
            debug_print(f"キャッシュからモデルを取得 / Getting model from cache", f"{model_type} [ID: {model_id}]", Color.CYAN)
        
        return self.models[model_id]
    
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
        
        if prompt_id not in self.prompts:
            if self.verbose:
                debug_print(f"プロンプトテンプレートを新規作成 / Creating new prompt template", f"[ID: {prompt_id}]", Color.CYAN)
            self.prompts[prompt_id] = ChatPromptTemplate.from_messages(messages)
        elif self.verbose:
            debug_print(f"キャッシュからプロンプトテンプレートを取得 / Getting prompt template from cache", f"[ID: {prompt_id}]", Color.CYAN)
        
        return self.prompts[prompt_id]

# レジストリのインスタンスを作成
# Create registry instance
registry = ComponentRegistry()
