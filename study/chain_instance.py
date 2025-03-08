"""
チェインごとにインスタンスカウンターを保持するモジュール
Module that maintains instance counters for each chain
"""

from langchain_core.runnables import RunnableLambda
import sys
import os
from typing import Dict, Any, Optional

# 現在のディレクトリの絶対パスを取得
# Get absolute path of current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# プロジェクトのルートディレクトリを取得
# Get project root directory
project_root = os.path.dirname(current_dir)
# パスを追加
# Add path
sys.path.append(project_root)

# snappychainパッケージをインポート
# Import snappychain package
from src.snappychain import set_chain_id, get_chain_id


# チェインカウンターを管理するグローバル辞書
# Global dictionary to manage chain counters
_chain_counters: Dict[str, int] = {}


def chain_counter() -> RunnableLambda:
    """
    カウンター機能を持つRunnableLambdaを作成する
    Create a RunnableLambda with counter functionality
    
    Returns:
        RunnableLambda: カウンター機能を持つRunnableLambda / RunnableLambda with counter functionality
    """
    def counter_func(x: Any) -> Dict[str, Any]:
        """
        カウンターをインクリメントして結果を返す
        Increment counter and return result
        """
        # 自身のチェインIDを取得
        # Get chain ID of this chain
        chain_id = get_chain_id(counter_lambda)
        
        # カウンターが存在しない場合は初期化
        # Initialize counter if it doesn't exist
        if chain_id not in _chain_counters:
            _chain_counters[chain_id] = 0
        
        # カウンターをインクリメント
        # Increment counter
        _chain_counters[chain_id] += 1
        
        return {
            "chain_id": chain_id,
            "counter": _chain_counters[chain_id]
        }
    
    # RunnableLambdaを作成
    # Create RunnableLambda
    counter_lambda = RunnableLambda(func=counter_func)
    
    # チェインIDを設定（snappychainの機能を使用）
    # Set chain ID (using snappychain functionality)
    set_chain_id(counter_lambda)
    
    return counter_lambda


if __name__ == "__main__":
    try:
        # カウンター機能を持つチェインを作成
        # Create a chain with counter functionality
        print("チェインを作成します / Creating chain...")
        chain = chain_counter()
        
        # チェインIDを表示
        # Display chain ID
        print(f"Chain ID: {get_chain_id(chain)}")
        
        # 実行して結果を表示
        # Execute and display results
        print("\n最初の実行 / First execution:")
        result1 = chain.invoke("test")
        print(f"Chain ID: {result1['chain_id']}")
        print(f"カウンター / Counter: {result1['counter']}")
        
        print("\n2回目の実行 / Second execution:")
        result2 = chain.invoke("test")
        print(f"Chain ID: {result2['chain_id']}")
        print(f"カウンター / Counter: {result2['counter']}")
        
        # 別のチェインを作成して実行
        # Create and execute another chain
        print("\n別のチェインの実行 / Execution of another chain:")
        another_chain = chain_counter()
        print(f"Chain ID: {get_chain_id(another_chain)}")
        result3 = another_chain.invoke("test")
        print(f"Chain ID: {result3['chain_id']}")
        print(f"カウンター / Counter: {result3['counter']}")
    except Exception as e:
        print(f"エラーが発生しました / An error occurred: {e}")
