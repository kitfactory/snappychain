from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.output_parsers.structured import StructuredOutputParser
from langchain_core.output_parsers import StrOutputParser


from onelogger import Logger

logger = Logger.get_logger(__name__)

def _get_template(data: dict) -> ChatPromptTemplate:
    """Create a ChatPromptTemplate from the provided data.

    Args:
        data (dict): The data containing the prompts.

    Raises:
        ValueError: If no prompts are found in the data.

    Returns:
        ChatPromptTemplate: The created ChatPromptTemplate.
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
    return ChatPromptTemplate.from_messages(messages)


def _chat(data:dict , model) -> dict:
    session = data["_session"]
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
            
            if data.get("_dev", False) == True:
                logger.debug("\033[32mAdded format instructions: %s\033[0m", format_instructions)
        except Exception as e:
            logger.error("\033[31mError setting up structured output: %s\033[0m", str(e))
    
    template = _get_template(data)
    template_replaced = template.invoke(data)

    # 開発モードの時はログ表示
    if data.get("_dev", False) == True:
        logger.debug("\033[32mLLM Request:%s\033[0m", template_replaced)

    # LLMに確認する
    response = model.invoke(template_replaced)
    if data.get("_dev", False) == True:
        logger.debug("\033[33mResponse:%s\033[0m", response)
    
    # 構造化出力のパースを試みる
    # Try to parse structured output
    if schemas:
        try:
            parsed_response = parser.parse(response.content)
            session["structured_response"] = parsed_response
            if data.get("_dev", False) == True:
                logger.debug("\033[33mStructured Response:%s\033[0m", parsed_response)
        except Exception as e:
            logger.warning("\033[31mFailed to parse structured response: %s\033[0m", str(e))
            # パースに失敗した場合でも元の応答は保存
            # Store the original response even if parsing fails
            
    session["response"] = response
    return data

def openai_chat(model="gpt-4o-mini", temperature=0.2) -> RunnableLambda:
    """
    Create a runnable lambda that generates a response using the OpenAI chat model following LangChain LCEL.
    LangChain LCELに沿ってOpenAIチャットモデルを使用し、応答を生成する実行可能なlambdaを返します。

    Returns:
        RunnableLambda: 実行可能なlambda関数 / Runnable lambda function.
    """
    def inner(data: dict) -> dict:
        """
        Extracts the last user prompt and generates a chat response using OpenAI's chat model.
        最後のユーザープロンプトを抽出し、OpenAIのチャットモデルを使用して応答を生成する内部関数です。

        Args:
            data (dict): 入力データ辞書 / Input data dictionary.

        Returns:
            dict: 応答が追加されたデータ辞書 / Data dictionary with the chat response appended.
        """
        # OpenAIチャットモデルを初期化（例としてgpt-3.5-turboとtemperature=0を使用）
        llm = ChatOpenAI(model_name=model, temperature=temperature)
        return _chat(data, llm)

    return RunnableLambda(inner)

# def gemini_chat() -> ChatOpenAI:
#     """
#     Google Gemini Chatモデルを返却する。
#     Returns a ChatOpenAI instance for Google Gemini configured with model_name 'googlegemini'.

#     Returns:
#         ChatOpenAI: Google Geminiチャットモデルのインスタンス / Instance of Google Gemini chat model.
#     """
#     return ChatOpenAI(model_name="googlegemini", temperature=0)

# def ollama_chat() -> ChatOllama:
#     """
#     Ollama Chatモデルを返却する。
#     Returns a ChatOllama instance.

#     Returns:
#         ChatOllama: Ollamaチャットモデルのインスタンス / Instance of Ollama chat model.
#     """
#     return ChatOllama()

# def anthropic_chat() -> ChatAnthropic:
#     """
#     Anthropic Chatモデルを返却する。
#     Returns a ChatAnthropic instance.

#     Returns:
#         ChatAnthropic: Anthropicチャットモデルのインスタンス / Instance of Anthropic chat model.
#     """
#     return ChatAnthropic()