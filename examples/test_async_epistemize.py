"""
非同期処理のテスト用スクリプト
Async processing test script
"""

import logging
import sys
import time
import json
from snappychain import epistemize, set_debug
from oneenv import load_dotenv

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()

def test_async_epistemize():
    """
    Test the asynchronous processing of epistemize function.
    epistemize関数の非同期処理をテストします。
    """
    logger.info("非同期処理テスト開始")
    
    # デバッグモードを有効化
    set_debug(True)
    
    # テスト用のデータを作成
    test_data = {
        "_session": {
            "response": {
                "content": """
                量子コンピュータは、量子力学の原理を利用して計算を行うコンピュータです。
                従来のコンピュータが0と1のビットを使用するのに対し、量子コンピュータは量子ビット（キュービット）を使用します。
                量子ビットは重ね合わせの原理により、0と1の状態を同時に取ることができます。
                これにより、特定の問題に対して指数関数的な高速化を実現できる可能性があります。
                量子コンピュータの応用分野としては、暗号解読、材料設計、機械学習、最適化問題などが挙げられます。
                """
            }
        }
    }
    
    # epistemize関数を実行
    logger.info("epistemize関数を実行中...")
    epistemize_fn = epistemize(model="gpt-4o-mini", temperature=0.2)
    result = epistemize_fn.invoke(test_data, verbose=True)
    
    # 初期状態を確認
    logger.info("初期状態（非同期処理開始直後）:")
    if "_session" in result and "epistemize" in result["_session"]:
        epistemize_data = result["_session"]["epistemize"]
        logger.info(f"処理中フラグ: {epistemize_data.get('processing', False)}")
        logger.info(f"新語数: {len(epistemize_data.get('neologisms', []))}")
        logger.info(f"実践数: {len(epistemize_data.get('praxis', []))}")
    else:
        logger.warning("epistemizeデータが見つかりません")
    
    # 非同期処理の完了を待つ
    logger.info("非同期処理の完了を待っています...")
    max_wait_time = 30  # 最大30秒待機
    wait_interval = 1.0  # 1秒ごとに確認
    
    for i in range(int(max_wait_time / wait_interval)):
        time.sleep(wait_interval)
        
        # セッションデータを再確認
        if "_session" in result and "epistemize" in result["_session"]:
            current_processing = result["_session"]["epistemize"].get("processing", False)
            if not current_processing:
                logger.info(f"処理が完了しました（{(i+1)*wait_interval}秒後）")
                break
        
        if i % 5 == 0:  # 5秒ごとに状態を出力
            logger.info(f"待機中... ({(i+1)*wait_interval}秒経過)")
    
    # 処理完了後の状態を確認
    logger.info("処理完了後の状態:")
    if "_session" in result and "epistemize" in result["_session"]:
        epistemize_data = result["_session"]["epistemize"]
        logger.info(f"処理中フラグ: {epistemize_data.get('processing', False)}")
        logger.info(f"新語数: {len(epistemize_data.get('neologisms', []))}")
        logger.info(f"実践数: {len(epistemize_data.get('praxis', []))}")
        
        if len(epistemize_data.get('neologisms', [])) > 0:
            logger.info("抽出された新語:")
            for item in epistemize_data["neologisms"]:
                logger.info(f"- {item['term']}: {item['definition']}")
        
        if len(epistemize_data.get('praxis', [])) > 0:
            logger.info("抽出された実践:")
            for item in epistemize_data["praxis"]:
                logger.info(f"- {item['instruction']} (適用文脈: {item['context']})")
    else:
        logger.warning("epistemizeデータが見つかりません")
    
    logger.info("テスト完了")

if __name__ == "__main__":
    test_async_epistemize()
