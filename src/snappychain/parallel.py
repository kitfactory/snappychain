"""
チェインを並列実行するための機能を提供するモジュール
Module providing functionality for parallel chain execution
"""

from typing import Dict, Any, Mapping, Optional, Union
from langchain_core.runnables import RunnableParallel, Runnable, RunnableConfig

def parallel(
    steps: Union[Dict[str, Runnable], Mapping[str, Runnable]], 
    config: Optional[RunnableConfig] = None
) -> RunnableParallel:
    """
    複数のチェインを並列実行するためのRunnableParallelを作成する
    Create a RunnableParallel for executing multiple chains in parallel

    Args:
        steps (Union[Dict[str, Runnable], Mapping[str, Runnable]]): 
            並列実行するチェインの辞書 / Dictionary of chains to execute in parallel
            キーは結果を格納する名前、値は実行するチェイン / Keys are names to store results, values are chains to execute
        config (Optional[RunnableConfig], optional): 
            実行時の設定 / Configuration for execution. Defaults to None.

    Returns:
        RunnableParallel: 並列実行可能なチェイン / Parallel executable chain

    Examples:
        >>> from langchain_core.prompts import ChatPromptTemplate
        >>> from langchain_openai import ChatOpenAI
        >>> 
        >>> # チェインを作成 / Create chains
        >>> model = ChatOpenAI()
        >>> joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
        >>> poem_chain = ChatPromptTemplate.from_template("write a poem about {topic}") | model
        >>> 
        >>> # 並列実行用のチェインを作成 / Create chain for parallel execution
        >>> parallel_chain = parallel({
        ...     "joke": joke_chain,
        ...     "poem": poem_chain
        ... })
        >>> 
        >>> # 実行 / Execute
        >>> result = parallel_chain.invoke({"topic": "python"})
        >>> print(result["joke"])
        >>> print(result["poem"])
    """
    return RunnableParallel(steps=steps)

def stream_parallel(
    steps: Union[Dict[str, Runnable], Mapping[str, Runnable]], 
    input_data: Dict[str, Any],
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    複数のチェインを並列実行し、結果をストリーミングする
    Execute multiple chains in parallel and stream the results

    Args:
        steps (Union[Dict[str, Runnable], Mapping[str, Runnable]]): 
            並列実行するチェインの辞書 / Dictionary of chains to execute in parallel
            キーは結果を格納する名前、値は実行するチェイン / Keys are names to store results, values are chains to execute
        input_data (Dict[str, Any]): 
            入力データ / Input data
        config (Optional[RunnableConfig], optional): 
            実行時の設定 / Configuration for execution. Defaults to None.

    Returns:
        Dict[str, Any]: ストリーミング結果の辞書 / Dictionary of streaming results

    Examples:
        >>> from langchain_core.prompts import ChatPromptTemplate
        >>> from langchain_openai import ChatOpenAI
        >>> 
        >>> # チェインを作成 / Create chains
        >>> model = ChatOpenAI()
        >>> joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
        >>> poem_chain = ChatPromptTemplate.from_template("write a poem about {topic}") | model
        >>> 
        >>> # 並列実行とストリーミング / Parallel execution and streaming
        >>> steps = {
        ...     "joke": joke_chain,
        ...     "poem": poem_chain
        ... }
        >>> 
        >>> # ストリーミング実行 / Streaming execution
        >>> output = stream_parallel(steps, {"topic": "python"})
        >>> print(output)
    """
    # RunnableParallelを作成 / Create RunnableParallel
    runnable = RunnableParallel(steps=steps)
    
    # 出力用の辞書を初期化 / Initialize output dictionary
    output = {key: "" for key in steps.keys()}
    
    # ストリーミング実行 / Streaming execution
    for chunk in runnable.stream(input_data, config=config):
        for key in chunk:
            if hasattr(chunk[key], "content"):
                output[key] = output[key] + chunk[key].content
            else:
                output[key] = output[key] + str(chunk[key])
                
    return output

async def astream_parallel(
    steps: Union[Dict[str, Runnable], Mapping[str, Runnable]], 
    input_data: Dict[str, Any],
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    複数のチェインを非同期で並列実行し、結果をストリーミングする
    Execute multiple chains in parallel asynchronously and stream the results

    Args:
        steps (Union[Dict[str, Runnable], Mapping[str, Runnable]]): 
            並列実行するチェインの辞書 / Dictionary of chains to execute in parallel
            キーは結果を格納する名前、値は実行するチェイン / Keys are names to store results, values are chains to execute
        input_data (Dict[str, Any]): 
            入力データ / Input data
        config (Optional[RunnableConfig], optional): 
            実行時の設定 / Configuration for execution. Defaults to None.

    Returns:
        Dict[str, Any]: ストリーミング結果の辞書 / Dictionary of streaming results

    Examples:
        >>> import asyncio
        >>> from langchain_core.prompts import ChatPromptTemplate
        >>> from langchain_openai import ChatOpenAI
        >>> 
        >>> async def main():
        ...     # チェインを作成 / Create chains
        ...     model = ChatOpenAI()
        ...     joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
        ...     poem_chain = ChatPromptTemplate.from_template("write a poem about {topic}") | model
        ...     
        ...     # 並列実行とストリーミング / Parallel execution and streaming
        ...     steps = {
        ...         "joke": joke_chain,
        ...         "poem": poem_chain
        ...     }
        ...     
        ...     # 非同期ストリーミング実行 / Asynchronous streaming execution
        ...     output = await astream_parallel(steps, {"topic": "python"})
        ...     print(output)
        >>> 
        >>> asyncio.run(main())
    """
    # RunnableParallelを作成 / Create RunnableParallel
    runnable = RunnableParallel(steps=steps)
    
    # 出力用の辞書を初期化 / Initialize output dictionary
    output = {key: "" for key in steps.keys()}
    
    # 非同期ストリーミング実行 / Asynchronous streaming execution
    async for chunk in runnable.astream(input_data, config=config):
        for key in chunk:
            if hasattr(chunk[key], "content"):
                output[key] = output[key] + chunk[key].content
            else:
                output[key] = output[key] + str(chunk[key])
                
    return output 