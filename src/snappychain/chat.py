from langchain_core.runnables import RunnableLambda, Runnable
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.output_parsers.structured import StructuredOutputParser
from langchain_core.output_parsers import StrOutputParser

from .print import debug_print, debug_request, debug_response, debug_error, Color, is_verbose, verbose_print
from .registry import registry
from .chain import get_chain_id, get_step_index

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
    # verboseパラメータを取得（kwargsまたはLangChainのグローバル設定から）
    # Get verbose parameter (from kwargs or LangChain global settings)
    verbose_param = kwargs.get("verbose", False)
    
    # kwargsのverboseまたはLangChainのverbose設定のいずれかがTrueの場合、verboseモードを有効にする
    # Enable verbose mode if either kwargs verbose or LangChain verbose setting is True
    verbose_mode = verbose_param or is_verbose()
    
    if verbose_mode:
        # verboseモードが有効な場合はデータを表示
        # Display data if verbose mode is enabled
        debug_request(data)
    
    # レジストリのverboseモードを設定
    # Set verbose mode for registry
    registry.set_verbose(verbose_mode)
    
    session = data["_session"]
    
    # セッションにargsとkwargsを保存
    # Save args and kwargs to session
    if args:
        # タプルとして保存（辞書ではない）
        # Save as a tuple (not a dictionary)
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
            model = ChatOllama(model=kwargs.get("model_name", "llama3"), 
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

    # verboseモードの時はログ表示
    if verbose_mode:
        # LLMに入力する最終的なプロンプト文字列はYELLOW
        verbose_print("LLMリクエスト / LLM Request", template_replaced, Color.YELLOW)

    # LLMに確認する
    response = model.invoke(template_replaced)
    if verbose_mode:
        # LLMからの返答や出力はGREEN
        verbose_print("LLM応答 / LLM Response", response, Color.GREEN)
    
    schemas = session.get("schema", [])
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
            
            if verbose_mode:
                # VectorSearchなど各Retriver結果はCYAN
                verbose_print("フォーマット指示 / Format Instructions", format_instructions, Color.CYAN)
                
            # 構造化出力のパースを試みる
            # Try to parse structured output
            try:
                parsed_response = parser.parse(response.content)
                session["structured_response"] = parsed_response
                if verbose_mode:
                    # Rerank結果はMAGENTA
                    verbose_print("構造化応答 / Structured Response", parsed_response, Color.MAGENTA)
            except Exception as e:
                debug_error(f"構造化応答のパース失敗 / Failed to parse structured response: {str(e)}")
                # パースに失敗した場合でも元の応答は保存
                # Store the original response even if parsing fails
        except Exception as e:
            debug_error(f"構造化出力の設定エラー / Error setting up structured output: {str(e)}")
    
    session["response"] = response
    return data

def openai_chat(model="gpt-4o-mini", temperature=0.2) -> RunnableLambda:
    """
    Create a runnable lambda that generates a response using the OpenAI chat model following LangChain LCEL.
    LangChain LCELに沿ってOpenAIチャットモデルを使用し、応答を生成する実行可能なlambdaを返します。

    Args:
        model (str): 使用するモデル名 / Model name to use
        temperature (float): 温度パラメータ / Temperature parameter

    Returns:
        RunnableLambda: 実行可能なlambda関数 / Runnable lambda function.
    """
    def inner(data: dict, *args, **kwargs) -> dict:
        """
        Extracts the last user prompt and generates a chat response using OpenAI's chat model.
        最後のユーザープロンプトを抽出し、OpenAIのチャットモデルを使用して応答を生成する内部関数です。

        Args:
            data (dict): 入力データ辞書 / Input data dictionary.
            *args: 可変長位置引数 / Variable length positional arguments
            **kwargs: 追加のパラメータ（verboseなど） / Additional parameters (e.g., verbose)

        Returns:
            dict: 応答が追加されたデータ辞書 / Data dictionary with the chat response appended.
        """
        # モデルパラメータを設定
        # Set model parameters
        model_kwargs = {
            "model_name": model,
            "temperature": temperature
        }
        
        # セッションにargsとkwargsを保存
        # Save args and kwargs to session
        if "_session" not in data:
            data["_session"] = {}
        session = data["_session"]
        
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
        
        # _chat関数を呼び出し
        # Call _chat function
        return _chat(data, "openai", *args, **{**model_kwargs, **kwargs})

    return RunnableLambda(inner)

def anthropic_chat(model="claude-3-haiku-20240307", temperature=0.2) -> RunnableLambda:
    """
    Create a runnable lambda that generates a response using the Anthropic chat model following LangChain LCEL.
    LangChain LCELに沿ってAnthropicチャットモデルを使用し、応答を生成する実行可能なlambdaを返します。

    Args:
        model (str): 使用するモデル名 / Model name to use
        temperature (float): 温度パラメータ / Temperature parameter

    Returns:
        RunnableLambda: 実行可能なlambda関数 / Runnable lambda function.
    """
    def inner(data: dict, *args, **kwargs) -> dict:
        """
        Extracts the last user prompt and generates a chat response using Anthropic's chat model.
        最後のユーザープロンプトを抽出し、Anthropicのチャットモデルを使用して応答を生成する内部関数です。

        Args:
            data (dict): 入力データ辞書 / Input data dictionary.
            *args: 可変長位置引数 / Variable length positional arguments
            **kwargs: 追加のパラメータ（verboseなど） / Additional parameters (e.g., verbose)

        Returns:
            dict: 応答が追加されたデータ辞書 / Data dictionary with the chat response appended.
        """
        # モデルパラメータを設定
        # Set model parameters
        model_kwargs = {
            "model_name": model,
            "temperature": temperature
        }
        
        # セッションにargsとkwargsを保存
        # Save args and kwargs to session
        if "_session" not in data:
            data["_session"] = {}
        session = data["_session"]
        
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
        
        # _chat関数を呼び出し
        # Call _chat function
        return _chat(data, "anthropic", *args, **{**model_kwargs, **kwargs})

    return RunnableLambda(inner)

def ollama_chat(model="llama3", temperature=0.2) -> RunnableLambda:
    """
    Create a runnable lambda that generates a response using the Ollama chat model following LangChain LCEL.
    LangChain LCELに沿ってOllamaチャットモデルを使用し、応答を生成する実行可能なlambdaを返します。

    Args:
        model (str): 使用するモデル名 / Model name to use
        temperature (float): 温度パラメータ / Temperature parameter

    Returns:
        RunnableLambda: 実行可能なlambda関数 / Runnable lambda function.
    """
    def inner(data: dict, *args, **kwargs) -> dict:
        """
        Extracts the last user prompt and generates a chat response using Ollama's chat model.
        最後のユーザープロンプトを抽出し、Ollamaのチャットモデルを使用して応答を生成する内部関数です。

        Args:
            data (dict): 入力データ辞書 / Input data dictionary.
            *args: 可変長位置引数 / Variable length positional arguments
            **kwargs: 追加のパラメータ（verboseなど） / Additional parameters (e.g., verbose)

        Returns:
            dict: 応答が追加されたデータ辞書 / Data dictionary with the chat response appended.
        """
        # モデルパラメータを設定
        # Set model parameters
        model_kwargs = {
            "model_name": model,
            "temperature": temperature
        }
        
        # セッションにargsとkwargsを保存
        # Save args and kwargs to session
        if "_session" not in data:
            data["_session"] = {}
        session = data["_session"]
        
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
        
        # _chat関数を呼び出し
        # Call _chat function
        return _chat(data, "ollama", *args, **{**model_kwargs, **kwargs})

    return RunnableLambda(inner)