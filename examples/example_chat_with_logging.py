"""
ロギングと詳細な結果表示を含むチャット例
Chat example with logging and detailed result display
"""

from snappychain import system_prompt, human_prompt
from snappychain import openai_chat, output
from snappychain import schema
from oneenv import load_dotenv
import logging
import json
import os
from datetime import datetime

# ロギングの設定
# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"chain_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# コンソールにも出力
# Also output to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

from langchain.globals import set_verbose

load_dotenv()

set_verbose(True)

def save_result_to_file(result, filename="result.json"):
    """
    結果をJSONファイルに保存する
    Save result to JSON file
    
    Args:
        result: 保存する結果 / Result to save
        filename: ファイル名 / Filename
    """
    # 結果を辞書に変換
    # Convert result to dictionary
    result_dict = {}
    
    if isinstance(result, dict):
        result_dict = result
    else:
        result_dict = {"result": str(result)}
    
    # セッション情報を抽出
    # Extract session information
    if "_session" in result_dict:
        session = result_dict["_session"]
        
        # モデルとレスポンスを文字列に変換
        # Convert model and response to string
        if "model" in session:
            session["model"] = str(session["model"])
        if "response" in session:
            session["response"] = str(session["response"])
    
    # JSONファイルに保存
    # Save to JSON file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        logging.info(f"結果を{filename}に保存しました / Result saved to {filename}")
    except Exception as e:
        logging.error(f"ファイル保存エラー / File save error: {str(e)}")

def print_structured_data(result):
    """
    構造化データを表示する
    Display structured data
    
    Args:
        result: 結果データ / Result data
    """
    if isinstance(result, dict) and "_session" in result:
        session = result["_session"]
        if "structured_response" in session:
            structured_data = session["structured_response"]
            print("\n構造化データ / Structured data:")
            for key, value in structured_data.items():
                print(f"{key}: {value}")

# メイン関数
# Main function
def main():
    logging.info("チェイン実行開始 / Chain execution started")
    
    try:
        # チェインを作成
        # Create a chain
        chain = system_prompt("あなたは有能なアシスタントです。") \
            | human_prompt("{question}") \
            | schema([
                {
                    "name":"場所",
                    "description": "山の場所"
                },
                {
                    "name":"標高",
                    "description": "山の標高(m)"
                }
            ])\
            | openai_chat(model="gpt-4o-mini", temperature=0.2) \
            | output()
        
        logging.info("チェインを作成しました / Chain created")
        
        # 基本的な呼び出し
        # Basic call
        question = "富士山について教えて"
        logging.info(f"質問: {question}")
        
        result = chain.invoke({"question": question})
        
        logging.info("チェイン実行完了 / Chain execution completed")
        
        # 結果の一部を表示
        # Display part of the result
        print("応答結果（一部） / Response result (partial):")
        if isinstance(result, dict) and "_session" in result and "response" in result["_session"]:
            response = result["_session"]["response"]
            content = getattr(response, "content", str(response))
            print(content)
        else:
            print(str(result)[:1000] + "..." if len(str(result)) > 1000 else str(result))
        
        # 構造化データを表示
        # Display structured data
        print_structured_data(result)
        
        # 結果をファイルに保存
        # Save result to file
        save_result_to_file(result)
        
    except Exception as e:
        logging.error(f"エラーが発生しました / An error occurred: {str(e)}", exc_info=True)
        print(f"エラーが発生しました。詳細はログファイル {log_file} を確認してください。")
        print(f"An error occurred. See log file {log_file} for details.")

if __name__ == "__main__":
    main()
    print(f"ログファイル: {log_file}")
    print(f"Log file: {log_file}")
