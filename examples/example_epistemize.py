from snappychain import system_prompt, human_prompt, openai_chat, epistemize, output, set_debug
from oneenv import load_dotenv
import traceback
import json
import logging
import sys
import time

# ロギングの設定
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv()

def main():
    """
    Example of using the epistemize function to extract neologisms and praxis from chat responses.
    チャットレスポンスから新語（Neologism）と実践（Praxis）を抽出するepistemize関数の使用例。
    """
    try:
        logging.debug("ステップ1: モジュールのインポート確認")
        
        logging.debug("ステップ2: チャットチェインの作成")
        # デバッグモードを有効化
        set_debug(True)
        chat_chain = system_prompt("あなたは有能なAIアシスタントです。専門用語を使って詳しく説明してください。") \
            | human_prompt("{question}") \
            | openai_chat(model="gpt-4o-mini", temperature=0.2)
        
        logging.debug("ステップ3: epistemizeチェインの作成")
        knowledge_chain = chat_chain | epistemize() | output()
        
        logging.debug("ステップ4: チェインの実行")
        result = knowledge_chain.invoke({"question": "量子コンピュータの基本原理と応用について教えてください"}, verbose=True)
        
        logging.debug("ステップ5: 結果の構造を確認")
        logging.debug(f"result型: {type(result)}")
        
        if isinstance(result, dict):
            logging.debug("resultは辞書型です")
            for key in result.keys():
                logging.debug(f"キー: {key}")
                
            if "_session" in result:
                session = result["_session"]
                logging.debug(f"_session型: {type(session)}")
                
                if isinstance(session, dict):
                    logging.debug("_sessionは辞書型です")
                    for key in session.keys():
                        logging.debug(f"  _sessionのキー: {key}")
                    
                    # 非同期処理の状態を確認
                    if "epistemize" in session:
                        epistemize_data = session["epistemize"]
                        logging.debug(f"  epistemize型: {type(epistemize_data)}")
                        
                        if isinstance(epistemize_data, dict):
                            logging.debug("  epistemizeは辞書型です")
                            
                            # 初期状態を確認
                            processing = epistemize_data.get("processing", False)
                            logging.debug(f"  処理中フラグ: {processing}")
                            
                            # 非同期処理の完了を待つ
                            if processing:
                                logging.debug("  非同期処理の完了を待っています...")
                                max_wait_time = 10  # 最大10秒待機
                                wait_interval = 0.5  # 0.5秒ごとに確認
                                
                                for i in range(int(max_wait_time / wait_interval)):
                                    time.sleep(wait_interval)
                                    
                                    # セッションデータを再確認
                                    current_processing = result["_session"]["epistemize"].get("processing", False)
                                    if not current_processing:
                                        logging.debug(f"  処理が完了しました（{(i+1)*wait_interval}秒後）")
                                        break
                                    
                                    if i % 2 == 0:  # 1秒ごとに状態を出力
                                        logging.debug(f"  待機中... ({(i+1)*wait_interval}秒経過)")
                                
                                # 最終状態を確認
                                final_epistemize_data = result["_session"]["epistemize"]
                                logging.debug(f"  最終処理中フラグ: {final_epistemize_data.get('processing', False)}")
                            
                            # 結果を表示
                            for key in epistemize_data.keys():
                                logging.debug(f"    epistemizeのキー: {key}")
                                logging.debug(f"    値の型: {type(epistemize_data[key])}")
                                
                                # 新語と実践の詳細を表示
                                if key == "neologisms" and isinstance(epistemize_data[key], list):
                                    logging.debug(f"    新語数: {len(epistemize_data[key])}")
                                    for item in epistemize_data[key]:
                                        if isinstance(item, dict) and "term" in item and "definition" in item:
                                            logging.debug(f"      - {item['term']}: {item['definition']}")
                                
                                elif key == "praxis" and isinstance(epistemize_data[key], list):
                                    logging.debug(f"    実践数: {len(epistemize_data[key])}")
                                    for item in epistemize_data[key]:
                                        if isinstance(item, dict) and "instruction" in item and "context" in item:
                                            logging.debug(f"      - {item['instruction']} (適用文脈: {item['context']})")
                                else:
                                    logging.debug(f"    値: {epistemize_data[key]}")
                        else:
                            logging.debug(f"  epistemizeは辞書型ではありません: {epistemize_data}")
                    else:
                        logging.debug("  epistemizeキーが存在しません")
                else:
                    logging.debug(f"_sessionは辞書型ではありません: {session}")
            else:
                logging.debug("_sessionキーが存在しません")
        else:
            logging.debug(f"resultは辞書型ではありません: {result}")
            
    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")
        logging.error("詳細なエラー情報:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
