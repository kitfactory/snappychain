from typing import Union, Any
from langchain_core.runnables import RunnableLambda
from .chain import Chain

SnappyChainData = Union[ str, dict ]

def _to_snappy(data: SnappyChainData)->dict:
    if isinstance(data, str):  # Check if data is of type str
        data={
            "_invoke_str": data,
            "_session" : {}
        }
    else:
        if "_session" not in data:
            data["_session"] = {}
    return data


def system_prompt(prompt: str) -> Chain:
    """
    システムプロンプトを作成する
    Create a system prompt
    
    Args:
        prompt (str): プロンプトの内容 / Prompt content
        
    Returns:
        Chain: チェインで使用できるChain / Chain that can be used in a chain
    """
    def inner(data: SnappyChainData, *args, **kwargs) -> dict:
        data = _to_snappy(data)
        session = data["_session"]
        
        # verboseパラメータを取得
        # Get verbose parameter
        verbose_mode = kwargs.get("verbose", False)
        
        # プロンプトを追加
        # Add prompt
        if "prompt" not in session:
            session["prompt"] = []
        session["prompt"].append({"system": prompt})
        
        # argsとkwargsをセッションに保存
        # Save args and kwargs to session
        if args:
            session["args"] = args
        if kwargs:
            session["kwargs"] = kwargs
            
        return data
    
    return Chain(inner)

def human_prompt(prompt: str) -> Chain:
    """
    ユーザープロンプトを作成する
    Create a human prompt
    
    Args:
        prompt (str): プロンプトの内容 / Prompt content
        
    Returns:
        Chain: チェインで使用できるChain / Chain that can be used in a chain
    """
    def inner(data: SnappyChainData, *args, **kwargs) -> dict:
        data = _to_snappy(data)
        session = data["_session"]
        
        # verboseパラメータを取得
        # Get verbose parameter
        verbose_mode = kwargs.get("verbose", False)
        
        # プロンプトを追加
        # Add prompt
        if "prompt" not in session:
            session["prompt"] = []
            
        # プロンプトのフォーマット
        # Format prompt
        if isinstance(data, dict):
            formatted_prompt = prompt.format(**data)
        else:
            formatted_prompt = prompt
            
        session["prompt"].append({"human": formatted_prompt})
        
        # argsとkwargsをセッションに保存
        # Save args and kwargs to session
        if args:
            session["args"] = args
        if kwargs:
            session["kwargs"] = kwargs
            
        return data
    
    return Chain(inner)


def ai_prompt(prompt: str) -> Chain:
    """
    AIプロンプトを作成する
    Create an AI prompt
    
    Args:
        prompt (str): プロンプトの内容 / Prompt content
        
    Returns:
        Chain: チェインで使用できるChain / Chain that can be used in a chain
    """
    def inner(data: SnappyChainData, *args, **kwargs) -> dict:
        data = _to_snappy(data)
        session = data["_session"]
        
        # verboseパラメータを取得
        # Get verbose parameter
        verbose_mode = kwargs.get("verbose", False)
        
        # プロンプトを追加
        # Add prompt
        if "prompt" not in session:
            session["prompt"] = []
            
        # プロンプトのフォーマット
        # Format prompt
        if isinstance(data, dict):
            formatted_prompt = prompt.format(**data)
        else:
            formatted_prompt = prompt
            
        session["prompt"].append({"ai": formatted_prompt})
        
        # argsとkwargsをセッションに保存
        # Save args and kwargs to session
        if args:
            session["args"] = args
        if kwargs:
            session["kwargs"] = kwargs
            
        return data
    
    return Chain(inner)