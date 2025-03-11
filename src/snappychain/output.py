"""
出力パーサーを提供するモジュール
Module providing output parsers
"""

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.list import MarkdownListOutputParser, NumberedListOutputParser
from langchain_core.runnables import RunnableLambda
from .print import debug_print, verbose_print, Color
from .registry import ComponentRegistry

# レジストリのインスタンスを取得
# Get registry instance
registry = ComponentRegistry()

def output(type: str = "text") -> RunnableLambda:
    """
    OutputParserを使用して応答を解析し、解析された値を直接返却します。
    Parse the response using OutputParser and return the parsed value directly.

    Args:
        type (str, optional): 
            出力形式を指定します。デフォルトは"text"です。
            Specify the output format. Defaults to "text".
            
            利用可能な形式 / Available formats:
            - text: テキスト形式 / Text format
            - json: JSON形式 / JSON format
            - markdown: Markdown形式 / Markdown format
            - numbered: 番号付きリスト形式 / Numbered list format

    Returns:
        RunnableLambda: 解析された値を返すRunnableLambda
                       RunnableLambda that returns the parsed value

    Examples:
        >>> from langchain_core.prompts import ChatPromptTemplate
        >>> from langchain_openai import ChatOpenAI
        >>> 
        >>> # チェインを作成 / Create chain
        >>> model = ChatOpenAI()
        >>> prompt = ChatPromptTemplate.from_template("数字を3つ挙げてください / List three numbers")
        >>> chain = prompt | model | output("numbered")
        >>> 
        >>> # 実行 / Execute
        >>> result = chain.invoke({})
        >>> print(result)  # [1, 2, 3]
    """
    # レジストリからパーサーを取得または作成
    # Get or create parser from registry
    parser_key = f"output_parser_{type}"
    parser = registry.get_object(parser_key)
    
    if parser is None:
        if type == "text":
            parser = StrOutputParser()
        elif type == "json":
            parser = JsonOutputParser()
        elif type == "markdown":
            parser = MarkdownListOutputParser()
        elif type == 'numbered':
            parser = NumberedListOutputParser()
        else:
            raise ValueError(f"Invalid output type: {type}")  # サポートされていない型の場合に例外を送出する / Raise error for unsupported type
        
        # 作成したパーサーをレジストリに保存
        # Save created parser to registry
        registry.set_object(parser_key, parser)

    def inner(data: dict, *args, **kwargs):
        """
        入力データから応答を取得し、パースして返却する内部関数
        Inner function to get response from input data, parse it and return
        
        Args:
            data (dict): 入力データ / Input data
            *args: 可変長位置引数 / Variable length positional arguments
            **kwargs: キーワード引数 / Keyword arguments
            
        Returns:
            Any: パースされた値 / Parsed value
        """
        # セッションを初期化
        # Initialize session
        if "_session" not in data:
            data["_session"] = {}
            
        session = data["_session"]
        
        # argsとkwargsをセッションに保存
        # Save args and kwargs to session
        if args:
            session["args"] = args
            
        # 既存のkwargsと新しいkwargsをマージ
        # Merge existing kwargs with new kwargs
        if "kwargs" not in session:
            session["kwargs"] = {}
            
        if not isinstance(session["kwargs"], dict):
            session["kwargs"] = {}
            
        session["kwargs"].update(kwargs)
        
        # structured_responseが存在する場合はそれを返す
        # If structured_response exists, return it
        if "structured_response" in session:
            structured_response = session["structured_response"]
            if kwargs.get("verbose") == True or session.get("kwargs", {}).get("verbose") == True:
                verbose_print("構造化レスポンスを返します / Returning structured response", 
                           f"{structured_response}", Color.YELLOW)
            return structured_response
        
        # 通常の応答処理
        # Normal response processing
        if isinstance(data, dict) and "_session" in data and "response" in data["_session"]:
            response = data["_session"]["response"]
        elif hasattr(data, "content"):
            response = data.content
        else:
            response = str(data)
            
        # 応答をパースして返却
        # Parse response and return
        parsed_response = parser.invoke(response)
        if kwargs.get("verbose") == True or session.get("kwargs", {}).get("verbose") == True:
            verbose_print("出力 / Output", f"{parsed_response}", Color.YELLOW)
        return parsed_response
    
    return RunnableLambda(inner)
