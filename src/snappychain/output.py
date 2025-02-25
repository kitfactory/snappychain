from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.list import MarkdownListOutputParser, NumberedListOutputParser
from langchain_core.runnables import RunnableLambda
from onelogger import Logger

logger = Logger.get_logger(__name__)

def output(type: str = "text") -> RunnableLambda:
    """
    Converts the string in the 'response' field of the input dictionary to a parsed output  
    and stores it using the key 'output'. Depending on the as_session flag, either the original  
    data dictionary (with the new 'output' key) or just the parsed output is returned.  
    入力辞書の 'response' キーにある文字列を解析し、その結果を 'output' キーに格納します。  
    as_session フラグにより、解析後の 'output' キーを含む元の辞書全体を返すか、  
    解析済みの応答のみを返すかが決まります。  

    The 'type' parameter supports the following formats:  
    type には以下の型を指定できます:  
    - text  
    - json  
    - markdown  
    - html  
    - latex  
    - python  

    Arguments:  
        response (str): The original response string contained in the input dictionary.  
                        入力辞書に格納された元の応答文字列。  
        as_session (bool): Flag indicating whether to return the entire data dictionary  
                           (including the parsed output under 'output') or just the parsed output.  
                           True の場合、データ全体を返し、それ以外の場合は解析済み応答のみを返します。  
    """
    parser = None
    if type == "text":
        parser = StrOutputParser()
    elif type == "json":
        parser = JsonOutputParser()
    elif type == "markdown":
        parser = MarkdownListOutputParser()
    elif type == 'numbered':
        parser = NumberedListOutputParser()
    # elif type == "html":
    #     parser = HtmlOutputParser()
    # elif type == "latex":
    #     parser = LatexOutputParser()
    # elif type == "python":
    #     parser = PythonOutputParser()
    else:
        raise ValueError(f"Invalid output type: {type}")  # Raise error for unsupported type | サポートされていない型の場合に例外を送出する

    # Inner function to parse the 'response' field and update the data dictionary with the parsed output.  
    # 応答辞書の 'response' キーの文字列を解析し、その結果を 'output' キーに格納する内部関数。
    def inner(data):
        session = data["_session"]
        
        # structured_responseが存在する場合はそれを返す
        # If structured_response exists, return it
        if "structured_response" in session:
            structured_response = session["structured_response"]
            if data.get("_dev") == True:
                logger.debug("\033[33mReturning structured response: %s\033[0m", structured_response)
            return structured_response
        
        # 通常の応答処理
        # Normal response processing
        response = session["response"]
        parsed_response = parser.invoke(response)
        if data.get("_dev") == True:
            logger.debug("\033[33mOutput: %s\033[0m", parsed_response)
        return parsed_response
    
    return RunnableLambda(inner)
