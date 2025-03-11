from langchain_core.runnables import RunnableLambda, Runnable
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.output_parsers.structured import StructuredOutputParser
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, Optional, List

from .print import verbose_print, Color
from .registry import registry
from .chain import get_chain_id, get_step_index, Chain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def _get_template(data: dict) -> ChatPromptTemplate:
    """Create a ChatPromptTemplate from the provided data.
    提供されたデータからChatPromptTemplateを作成します。

    Args:
        data (dict): The data containing the prompts.
                    プロンプトを含むデータ。

    Raises:
        ValueError: If no prompts are found in the data.
                   データにプロンプトが見つからない場合。

    Returns:
        ChatPromptTemplate: The created ChatPromptTemplate.
                           作成されたChatPromptTemplate。
    """
    session = data.get("_session", {})
    prompts = session.get("prompt", [])
    if prompts is None or len(prompts) == 0:
        raise ValueError("No prompts found in data")
    
    # チェインIDとインデックスを取得
    # Get chain ID and index
    chain_id = None
    index = 0
    
    # 現在の実行コンテキストからチェインIDとインデックスを取得
    # Get chain ID and index from current execution context
    if "chain" in session:
        chain = session.get("chain")
        chain_id = get_chain_id(chain)
        index = get_step_index(chain)
    
    # チェインIDが存在する場合、レジストリからテンプレートを取得
    # If chain ID exists, get template from registry
    if chain_id is not None:
        template = registry.get_chain_object(chain_id, index, "template")
        if template is not None:
            return template
    
    # テンプレートが存在しない場合、新規作成
    # If template does not exist, create a new one
    messages = []
    for p in prompts:
        if "system" in p:
            messages.append(SystemMessagePromptTemplate.from_template(p["system"]))
        if "human" in p:
            messages.append(HumanMessagePromptTemplate.from_template(p["human"]))
        if "ai" in p:
            messages.append(AIMessagePromptTemplate.from_template(p["ai"]))
    
    template = ChatPromptTemplate.from_messages(messages)
    
    # チェインIDが存在する場合、作成したテンプレートをレジストリに保存
    # If chain ID exists, save the created template to registry
    if chain_id is not None:
        registry.set_chain_object(chain_id, index, template, "template")
    
    return template


def _chat(data:dict, model_type, *args, **kwargs) -> dict:
    """
    LLMを使用してチャット応答を生成する内部関数
    Internal function to generate chat responses using an LLM
    
    Args:
        data (dict): 入力データ / Input data
        model_type (str): モデルの種類 / Model type
        *args: 可変長位置引数 / Variable length positional arguments
        **kwargs: キーワード引数 / Keyword arguments
        
    Returns:
        dict: 応答を含む更新されたデータ / Updated data with response
    """
    # verboseパラメータを取得
    # Get verbose parameter from kwargs
    verbose_mode = kwargs.get("verbose", False)
    
    if verbose_mode:
        verbose_print("デバッグ情報 / Debug info", {
            "入力データ / Input data": data,
            "モデル / Model": model_type,
            "引数 / Args": args,
            "キーワード引数 / Kwargs": kwargs
        }, Color.MAGENTA)
    
    # レジストリのverboseモードを設定
    # Set verbose mode for registry
    registry.set_verbose(verbose_mode)
    
    session = data["_session"]
    
    # セッションにargsとkwargsを保存
    # Save args and kwargs to session
    if args:
        session["args"] = args
    
    # 既存のkwargsと新しいkwargsをマージ
    # Merge existing kwargs with new kwargs
    session_kwargs = session.get("kwargs", {})
    if isinstance(session_kwargs, dict):
        session_kwargs.update(kwargs)
    else:
        session_kwargs = kwargs
    session["kwargs"] = session_kwargs
    
    # チェインIDとインデックスを取得
    # Get chain ID and index
    chain_id = None
    index = 0
    
    # 現在の実行コンテキストからチェインIDとインデックスを取得
    # Get chain ID and index from current execution context
    if "chain" in session:
        chain = session.get("chain")
        chain_id = get_chain_id(chain)
        index = get_step_index(chain)
    
    # モデルを取得
    # Get model
    model = None
    
    # チェインIDが存在する場合、レジストリからモデルを取得
    # If chain ID exists, get model from registry
    if chain_id is not None:
        model = registry.get_chain_object(chain_id, index, "model")
    
    # モデルが存在しない場合、新規作成
    # If model does not exist, create a new one
    if model is None:
        # モデルの種類に応じてインスタンスを作成
        # Create instance based on model type
        if model_type == "openai":
            model = ChatOpenAI(model_name=kwargs.get("model_name", "gpt-4o-mini"), 
                              temperature=kwargs.get("temperature", 0.7))
        elif model_type == "anthropic":
            model = ChatAnthropic(model_name=kwargs.get("model_name", "claude-3-haiku-20240307"), 
                                 temperature=kwargs.get("temperature", 0.7))
        elif model_type == "ollama":
            model = ChatOllama(model=kwargs.get("model", "llama3"), 
                              temperature=kwargs.get("temperature", 0.7))
        else:
            raise ValueError(f"不明なモデルタイプ / Unknown model type: {model_type}")
        
        # チェインIDが存在する場合、作成したモデルをレジストリに保存
        # If chain ID exists, save the created model to registry
        if chain_id is not None:
            registry.set_chain_object(chain_id, index, model, "model")
    
    session["model"] = model
    
    template = _get_template(data)
    template_replaced = template.invoke(data)

    if verbose_mode:
        # 1. チェインの開始を表示
        verbose_print("チェインの実行開始 / Chain execution start", 
                   f"Chain ID: {chain_id}, Model: {model_type}", Color.CYAN)
        
        # 2. リクエストの全体を表示（メッセージのみ）
        verbose_print("リクエスト / Request", [
            {"role": m.type_, "content": m.content} 
            for m in template_replaced.messages
        ], Color.GREEN)

    # LLMに確認する
    response = model.invoke(template_replaced)
    
    if verbose_mode:
        # 3. 応答アウトプットを表示
        verbose_print("応答 / Response", response.content, Color.YELLOW)
    
    schemas = session.get("schema", [])
    parsed_response = None
    if schemas:
        try:
            # 構造化出力用のパーサーを作成
            # Create parser for structured output
            parser = StructuredOutputParser.from_response_schemas(schemas)
            
            # パーサーのフォーマット手順を取得
            # Get formatting instructions from the parser
            format_instructions = parser.get_format_instructions()
            
            # 中括弧をエスケープ（二重中括弧にする）
            # Escape curly braces by doubling them
            format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
            
            # フォーマット手順をデータに追加
            # Add formatting instructions to data
            if "format_instructions" not in data:
                data["format_instructions"] = format_instructions
                
            # システムプロンプトに構造化出力の指示を追加
            # Add instructions for structured output to the system prompt
            for i, prompt in enumerate(session.get("prompt", [])):
                if "system" in prompt:
                    session["prompt"][i]["system"] += "\n\n" + format_instructions
                    break
            else:
                # システムプロンプトがない場合は最初に追加
                # If no system prompt exists, add it as the first one
                session["prompt"].insert(0, {"system": format_instructions})
            
            # 構造化出力のパースを試みる
            # Try to parse structured output
            try:
                parsed_response = parser.parse(response.content)
                session["structured_response"] = parsed_response
            except Exception as e:
                if verbose_mode:
                    verbose_print("エラー / Error", f"構造化応答のパース失敗 / Failed to parse structured response: {str(e)}", Color.RED)
                # パースに失敗した場合でも元の応答は保存
                # Store the original response even if parsing fails
        except Exception as e:
            if verbose_mode:
                verbose_print("エラー / Error", f"構造化出力の設定エラー / Error setting up structured output: {str(e)}", Color.RED)
    
    if verbose_mode:
        # 4. チェインの終了を表示（パース結果を含む）
        verbose_print("チェインの実行完了 / Chain execution complete", {
            "chain_id": chain_id,
            "result": parsed_response if parsed_response is not None else response.content
        }, Color.CYAN)
    
    session["response"] = response
    return data

def _convert_messages(messages: List[Dict[str, str]]) -> List[Any]:
    """
    Convert message dictionaries to LangChain message objects.
    メッセージ辞書をLangChainメッセージオブジェクトに変換します。

    Args:
        messages (List[Dict[str, str]]): List of message dictionaries.
                                       メッセージ辞書のリスト。

    Returns:
        List[Any]: List of LangChain message objects.
                  LangChainメッセージオブジェクトのリスト。
    """
    converted = []
    for msg in messages:
        if "system" in msg:
            converted.append(SystemMessage(content=msg["system"]))
        elif "human" in msg:
            converted.append(HumanMessage(content=msg["human"]))
        elif "ai" in msg:
            converted.append(AIMessage(content=msg["ai"]))
    return converted

def ollama_chat(model: str = "phi4-mini:latest", temperature: float = 0.2) -> Chain:
    """
    Create a chat chain using Ollama's models.
    Ollamaのモデルを使用してチャットチェーンを作成します。

    Args:
        model (str): The model to use. Defaults to "phi4-mini:latest".
                    使用するモデル。デフォルトは "phi4-mini:latest"。
        temperature (float): The temperature parameter. Defaults to 0.2.
                           温度パラメータ。デフォルトは0.2。

    Returns:
        Chain: A chain that can be used in a chain.
               チェーンで使用できるチェーン。
    """
    def inner(data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        if "_session" not in data:
            data["_session"] = {}
        
        # verboseパラメータを処理
        verbose_mode = kwargs.get("verbose", False)
        if verbose_mode:
            verbose_print("Ollama Chat", f"Starting chat with model: {model}", Color.CYAN)
        
        return _chat(data, "ollama", *args, **{
            "model": model,
            "temperature": temperature,
            "verbose": verbose_mode,
            **kwargs
        })
    
    return Chain(inner)

def openai_chat(model: str = "gpt-4o-mini", temperature: float = 0.2) -> Chain:
    """
    Create a chat chain using OpenAI's models.
    OpenAIのモデルを使用してチャットチェーンを作成します。

    Args:
        model (str): The model to use. Defaults to "gpt-4o-mini".
                    使用するモデル。デフォルトは "gpt-4o-mini"。
        temperature (float): The temperature parameter. Defaults to 0.2.
                           温度パラメータ。デフォルトは0.2。

    Returns:
        Chain: A chain that can be used in a chain.
               チェーンで使用できるチェーン。
    """
    def inner(data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        if "_session" not in data:
            data["_session"] = {}
        
        # verboseパラメータを処理
        verbose_mode = kwargs.get("verbose", False)
        if verbose_mode:
            verbose_print("OpenAI Chat", f"Starting chat with model: {model}", Color.CYAN)
        
        return _chat(data, "openai", *args, **{
            "model_name": model,
            "temperature": temperature,
            "verbose": verbose_mode,
            **kwargs
        })
    
    return Chain(inner)

def anthropic_chat(model: str = "claude-3-haiku-20240307", temperature: float = 0.2) -> Chain:
    """
    Create a chat chain using Anthropic's models.
    Anthropicのモデルを使用してチャットチェーンを作成します。

    Args:
        model (str): The model to use. Defaults to "claude-3-haiku-20240307".
                    使用するモデル。デフォルトは "claude-3-haiku-20240307"。
        temperature (float): The temperature parameter. Defaults to 0.2.
                           温度パラメータ。デフォルトは0.2。

    Returns:
        Chain: A chain that can be used in a chain.
               チェーンで使用できるチェーン。
    """
    def inner(data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        if "_session" not in data:
            data["_session"] = {}
        
        # verboseパラメータを処理
        verbose_mode = kwargs.get("verbose", False)
        if verbose_mode:
            verbose_print("Anthropic Chat", f"Starting chat with model: {model}", Color.CYAN)
        
        return _chat(data, "anthropic", *args, **{
            "model_name": model,
            "temperature": temperature,
            "verbose": verbose_mode,
            **kwargs
        })
    
    return Chain(inner)