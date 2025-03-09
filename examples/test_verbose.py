"""
verboseパラメータのテスト
Test for verbose parameter
"""

import sys
from snappychain import system_prompt, human_prompt
from snappychain import openai_chat, output
from snappychain.registry import ComponentRegistry
from oneenv import load_dotenv

load_dotenv()

# レジストリのverboseを有効化
# Enable verbose for registry
registry = ComponentRegistry()
registry.set_verbose(True)

# メイン関数
# Main function
def main():
    # チェインを作成
    # Create a chain
    chain = system_prompt("あなたは有能なアシスタントです。") \
        | human_prompt("{question}") \
        | openai_chat(model="gpt-4o-mini", temperature=0.2) \
        | output()
    
    # verboseなしの呼び出し
    # Call without verbose
    print("verboseなしの呼び出し / Call without verbose:", file=sys.stderr)
    result1 = chain.invoke({"question":"こんにちは、元気ですか？"})
    
    print("応答結果1 / Response result 1:", file=sys.stderr)
    print(result1, file=sys.stderr)
    print("-" * 50, file=sys.stderr)
    
    # verboseありの呼び出し
    # Call with verbose
    print("\nverboseありの呼び出し / Call with verbose:", file=sys.stderr)
    result2 = chain.invoke({"question":"今日の天気は？"}, verbose=True)
    
    print("応答結果2 / Response result 2:", file=sys.stderr)
    print(result2, file=sys.stderr)

if __name__ == "__main__":
    main()
