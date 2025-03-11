"""
並列実行機能のサンプルコード
Example code for parallel execution functionality
"""

import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from snappychain.parallel import parallel, stream_parallel, astream_parallel


def test_parallel():
    """
    parallel()関数のテスト
    Test parallel() function
    """
    print("\n=== parallel()のテスト / Testing parallel() ===")
    
    # チェインを作成 / Create chains
    model = ChatOpenAI()
    joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
    poem_chain = ChatPromptTemplate.from_template("write a poem about {topic}") | model
    
    # 並列実行用のチェインを作成 / Create chain for parallel execution
    parallel_chain = parallel({
        "joke": joke_chain,
        "poem": poem_chain
    })
    
    # 実行 / Execute
    result = parallel_chain.invoke({"topic": "python"})
    
    print("\nジョーク / Joke:")
    print(result["joke"])
    print("\n詩 / Poem:")
    print(result["poem"])


def test_stream_parallel():
    """
    stream_parallel()関数のテスト
    Test stream_parallel() function
    """
    print("\n=== stream_parallel()のテスト / Testing stream_parallel() ===")
    
    # チェインを作成 / Create chains
    model = ChatOpenAI()
    joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
    poem_chain = ChatPromptTemplate.from_template("write a poem about {topic}") | model
    
    # 並列実行とストリーミング / Parallel execution and streaming
    steps = {
        "joke": joke_chain,
        "poem": poem_chain
    }
    
    # ストリーミング実行 / Streaming execution
    output = stream_parallel(steps, {"topic": "python"})
    
    print("\nジョーク / Joke:")
    print(output["joke"])
    print("\n詩 / Poem:")
    print(output["poem"])


async def test_astream_parallel():
    """
    astream_parallel()関数のテスト
    Test astream_parallel() function
    """
    print("\n=== astream_parallel()のテスト / Testing astream_parallel() ===")
    
    # チェインを作成 / Create chains
    model = ChatOpenAI()
    joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
    poem_chain = ChatPromptTemplate.from_template("write a poem about {topic}") | model
    
    # 並列実行とストリーミング / Parallel execution and streaming
    steps = {
        "joke": joke_chain,
        "poem": poem_chain
    }
    
    # 非同期ストリーミング実行 / Asynchronous streaming execution
    output = await astream_parallel(steps, {"topic": "python"})
    
    print("\nジョーク / Joke:")
    print(output["joke"])
    print("\n詩 / Poem:")
    print(output["poem"])


def main():
    """
    メイン関数
    Main function
    """
    print("============================================================")
    print("=== 並列実行機能のテスト / Testing parallel execution features ===")
    print("============================================================")
    
    # 通常の並列実行をテスト / Test normal parallel execution
    test_parallel()
    
    # ストリーミング並列実行をテスト / Test streaming parallel execution
    test_stream_parallel()
    
    # 非同期ストリーミング並列実行をテスト / Test async streaming parallel execution
    asyncio.run(test_astream_parallel())
    
    print("\n============================================================")
    print("=== テスト完了 / Testing completed ===")
    print("============================================================")


if __name__ == "__main__":
    main() 