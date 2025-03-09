"""
argsとkwargsの伝達をテストするシンプルなスクリプト
Simple script to test args and kwargs transmission
"""

from snappychain import system_prompt, human_prompt, openai_chat, output
from oneenv import load_dotenv
import traceback
import sys

load_dotenv()

def main():
    """
    メイン関数
    Main function
    """
    try:
        print("シンプルなチェインを作成します / Creating a simple chain...")
        
        # シンプルなチェインを作成
        # Create a simple chain
        chain = system_prompt("あなたは有能なアシスタントです。") \
            | human_prompt("{question}") \
            | openai_chat(model="gpt-4o-mini", temperature=0.2) \
            | output()
        
        print("チェインを呼び出します（基本呼び出し） / Calling chain (basic call)...")
        
        # 基本的な呼び出し（追加パラメータなし）
        # Basic call (no additional parameters)
        result1 = chain.invoke({"question": "こんにちは、元気ですか？"})
        
        print("応答結果1 / Response result 1:")
        print(result1)
        
        print("\nチェインを呼び出します（kwargsのみ） / Calling chain (with kwargs only)...")
        
        # kwargsのみを使用した呼び出し
        # Call with kwargs only
        try:
            result2 = chain.invoke({"question": "今日の天気は？"}, verbose=True)
            print("応答結果2 / Response result 2:")
            print(result2)
        except Exception as e:
            print(f"kwargsを使用した呼び出しでエラーが発生しました / Error occurred with kwargs call: {str(e)}")
            traceback.print_exc(file=sys.stdout)  # 標準出力にエラー情報を出力
        
    except Exception as e:
        print(f"エラーが発生しました / An error occurred: {str(e)}")
        print("詳細なエラー情報 / Detailed error information:")
        traceback.print_exc(file=sys.stdout)  # 標準出力にエラー情報を出力

if __name__ == "__main__":
    main()
