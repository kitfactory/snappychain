from typing import Union, Any
from langchain_core.runnables import RunnableLambda

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


def system_prompt(prompt: str) -> RunnableLambda:
    """
    Execute system prompt.
    システムプロンプトを実行する。

    Args:
        prompt (str): System prompt content.
        prompt (str): システムプロンプトの内容。

    Returns:
        RunnableLambda: A lambda that takes a data dictionary, appends a system prompt entry to the 'prompt' array and returns the updated dictionary.
        RunnableLambda: data 辞書型を受け取り、'prompt' 配列にシステムプロンプトを追加して返却します。
    """
    def inner(data, *args, **kwargs):
        data = _to_snappy(data)
        # Ensure the 'prompt' key exists in the data dictionary, otherwise initialize it as an empty list.
        # データ辞書に 'prompt' キーが存在しない場合は、空のリストで初期化する。
        session = data["_session"]
        prompt_list = session.get("prompt", [])
        prompt_list.append({"system": prompt})
        session["prompt"] = prompt_list
        
        # argsとkwargsをセッションに保存して後続のチェインに伝達
        # Save args and kwargs to session to pass to subsequent chains
        if args:
            session["args"] = args
        if kwargs:
            session["kwargs"] = kwargs
            
        return data
    return RunnableLambda(inner)

def human_prompt(prompt: str) -> RunnableLambda:
    """
    Get or prompt user input.
    ユーザープロンプトを取得または促す。

    Args:
        prompt (str): User prompt content.
        prompt (str): ユーザープロンプトの内容。

    Returns:
        RunnableLambda: A lambda that takes a data dictionary, appends a user prompt entry to the 'prompt' array and returns the updated dictionary.
        RunnableLambda: data 辞書型を受け取り、'prompt' 配列にユーザープロンプトを追加して返却します。
    """
    def inner(data, *args, **kwargs):
        data = _to_snappy(data)
        session = data["_session"]
        prompt_list = session.get("prompt", [])
        prompt_list.append({"human": prompt})
        session["prompt"] = prompt_list
        
        # argsとkwargsをセッションに保存して後続のチェインに伝達
        # Save args and kwargs to session to pass to subsequent chains
        if args:
            session["args"] = args
        if kwargs:
            session["kwargs"] = kwargs
            
        return data
    return RunnableLambda(inner)


def ai_prompt(prompt: str) -> RunnableLambda:
    """
    Execute AI prompt.
    AI プロンプトを実行する。

    Args:
        prompt (str): AI prompt content.
        prompt (str): AI プロンプトの内容。

    Returns:
        RunnableLambda: A lambda that takes a data dictionary, appends an AI prompt entry to the 'prompt' array and returns the updated dictionary.
        RunnableLambda: data 辞書型を受け取り、'prompt' 配列に AI プロンプトを追加して返却します。
    """
    def inner(data, *args, **kwargs):
        data = _to_snappy(data)
        session = data["_session"]
        prompt_list = session.get("prompt", [])
        prompt_list.append({"ai": prompt})
        session["prompt"] = prompt_list
        
        # argsとkwargsをセッションに保存して後続のチェインに伝達
        # Save args and kwargs to session to pass to subsequent chains
        if args:
            session["args"] = args
        if kwargs:
            session["kwargs"] = kwargs
            
        return data
    return RunnableLambda(inner)