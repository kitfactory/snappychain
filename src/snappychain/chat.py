from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.output_parsers.structured import StructuredOutputParser
from langchain_core.output_parsers import StrOutputParser

from .print import debug_print, debug_request, debug_response, debug_error, Color, is_verbose
from .registry import registry

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
    messages = []
    for p in prompts:
        if "system" in p:
            messages.append(SystemMessagePromptTemplate.from_template(p["system"]))
        if "human" in p:
            messages.append(HumanMessagePromptTemplate.from_template(p["human"]))
        if "ai" in p:
            messages.append(AIMessagePromptTemplate.from_template(p["ai"]))
    
    # レジストリからプロンプトテンプレートを取得
    # Get prompt template from registry
    return registry.get_prompt(messages)


def _chat(data:dict, model_type, *args, **kwargs) -> dict:
    """
    LLMを使用してチャット応答を生成する内部関数
    Internal function to generate chat responses using an LLM
    
    Args:
        data (dict): 入力データ / Input data
        model_type (str): モデルの種類 / Model type
        
    Returns:
        dict: 応答を含む更新されたデータ / Updated data with response
    """
    # verboseパラメータを取得
    # Get verbose parameter
    verbose = kwargs.get("verbose", False)
    if verbose or data.get("_dev", False):
        # verboseモードが有効な場合はデータを表示
        # Display data if verbose mode is enabled
        debug_request(data)
    
    # レジストリのverboseモードを設定
    # Set verbose mode for registry
    registry.set_verbose(verbose or data.get("_dev", False))
    
    session = data["_session"]
    
    # レジストリからモデルを取得
    # Get model from registry
    model = registry.get_model(model_type, **kwargs)
    session["model"] = model
    
    # スキーマが指定されている場合、構造化出力用のパーサーを設定
    # If schema is specified, set up parser for structured output
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
            
            if verbose or data.get("_dev", False):
                debug_print("フォーマット指示 / Format Instructions", format_instructions, Color.CYAN)
        except Exception as e:
            debug_error(f"構造化出力の設定エラー / Error setting up structured output: {str(e)}")
    
    template = _get_template(data)
    template_replaced = template.invoke(data)

    # verboseモードの時はログ表示
    if verbose or data.get("_dev", False):
        debug_print("LLMリクエスト / LLM Request", template_replaced, Color.GREEN)

    # LLMに確認する
    response = model.invoke(template_replaced)
    if verbose or data.get("_dev", False):
        debug_print("LLM応答 / LLM Response", response, Color.YELLOW)
    
    # 構造化出力のパースを試みる
    # Try to parse structured output
    if schemas:
        try:
            parsed_response = parser.parse(response.content)
            session["structured_response"] = parsed_response
            if verbose or data.get("_dev", False):
                debug_print("構造化応答 / Structured Response", parsed_response, Color.MAGENTA)
        except Exception as e:
            debug_error(f"構造化応答のパース失敗 / Failed to parse structured response: {str(e)}")
            # パースに失敗した場合でも元の応答は保存
            # Store the original response even if parsing fails
            
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
    def inner(data: dict, **kwargs) -> dict:
        """
        Extracts the last user prompt and generates a chat response using OpenAI's chat model.
        最後のユーザープロンプトを抽出し、OpenAIのチャットモデルを使用して応答を生成する内部関数です。

        Args:
            data (dict): 入力データ辞書 / Input data dictionary.
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
        
        # _chat関数を呼び出し
        # Call _chat function
        return _chat(data, "openai", **model_kwargs, **kwargs)

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
    def inner(data: dict, **kwargs) -> dict:
        """
        Extracts the last user prompt and generates a chat response using Anthropic's chat model.
        最後のユーザープロンプトを抽出し、Anthropicのチャットモデルを使用して応答を生成する内部関数です。

        Args:
            data (dict): 入力データ辞書 / Input data dictionary.
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
        
        # _chat関数を呼び出し
        # Call _chat function
        return _chat(data, "anthropic", **model_kwargs, **kwargs)

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
    def inner(data: dict, **kwargs) -> dict:
        """
        Extracts the last user prompt and generates a chat response using Ollama's chat model.
        最後のユーザープロンプトを抽出し、Ollamaのチャットモデルを使用して応答を生成する内部関数です。

        Args:
            data (dict): 入力データ辞書 / Input data dictionary.
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
        
        # _chat関数を呼び出し
        # Call _chat function
        return _chat(data, "ollama", **model_kwargs, **kwargs)

    return RunnableLambda(inner)