"""
シンプルなテストスクリプト
Simple test script
"""

from snappychain import system_prompt, human_prompt
from snappychain import openai_chat, output
from oneenv import load_dotenv

load_dotenv()

# メイン関数
# Main function
def main():
    # チェインを作成
    # Create a chain
    chain = system_prompt("あなたは有能なアシスタントです。") \
        | human_prompt("{question}") \
        | openai_chat(model="gpt-4o-mini", temperature=0.2) \
        | output()
    
    # 基本的な呼び出し
    # Basic call
    result = chain.invoke({"question":"こんにちは、元気ですか？"})
    
    print("応答結果 / Response result:")
    print(result)
    
    # kwargsを使用した呼び出し
    # Call with kwargs
    print("\nkwargsを使用した呼び出し / Call with kwargs:")
    result2 = chain.invoke({"question":"今日の天気は？", "additional_info": "追加情報"}, verbose=True)
    
    print("応答結果2 / Response result 2:")
    print(result2)

if __name__ == "__main__":
    main()
