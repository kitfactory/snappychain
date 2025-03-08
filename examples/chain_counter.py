"""
チェインIDに対応したカウンターオブジェクトの例
Example of counter objects corresponding to chain IDs
"""

import sys
import os
from typing import Dict, Any, List

# snappychainパッケージからチェイン関連の機能をインポート
# Import chain related functionality from snappychain package
from snappychain.chain import generate_chain_id, get_chain_id


class ChainCounter:
    """
    チェインIDに対応したカウンターを管理するクラス
    Class to manage counters corresponding to chain IDs
    """
    def __init__(self):
        """
        初期化
        Initialization
        """
        # チェインIDをキー、カウンター値を値とする辞書
        # Dictionary with chain ID as key and counter value as value
        self.counters: Dict[str, int] = {}
        
    def increment(self, chain_id: str) -> int:
        """
        指定されたチェインIDのカウンターをインクリメントする
        Increment the counter for the specified chain ID
        
        Args:
            chain_id (str): チェインID / Chain ID
            
        Returns:
            int: インクリメント後のカウンター値 / Counter value after increment
        """
        if chain_id not in self.counters:
            self.counters[chain_id] = 0
        self.counters[chain_id] += 1
        return self.counters[chain_id]
    
    def get_count(self, chain_id: str) -> int:
        """
        指定されたチェインIDのカウンター値を取得する
        Get the counter value for the specified chain ID
        
        Args:
            chain_id (str): チェインID / Chain ID
            
        Returns:
            int: カウンター値 / Counter value
        """
        return self.counters.get(chain_id, 0)
    
    def get_all_counts(self) -> Dict[str, int]:
        """
        すべてのチェインIDのカウンター値を取得する
        Get counter values for all chain IDs
        
        Returns:
            Dict[str, int]: チェインIDとカウンター値の辞書 / Dictionary of chain IDs and counter values
        """
        return self.counters.copy()


# グローバルなカウンターインスタンス
# Global counter instance
counter = ChainCounter()


def main():
    """
    メイン関数
    Main function
    """
    print("=== チェインIDとカウンターの例 / Chain ID and Counter Example ===")
    
    # 異なるチェインIDを生成して使用
    # Generate and use different chain IDs
    chain_id1 = generate_chain_id()
    chain_id2 = generate_chain_id()
    
    print(f"\n--- チェインID1: {chain_id1} ---")
    for i in range(3):
        count = counter.increment(chain_id1)
        print(f"Increment {i+1}: Count = {count}")
    
    print(f"\n--- チェインID2: {chain_id2} ---")
    for i in range(2):
        count = counter.increment(chain_id2)
        print(f"Increment {i+1}: Count = {count}")
    
    # チェインID1を再度使用
    # Use chain ID1 again
    print(f"\n--- チェインID1の再使用 / Reuse of Chain ID1: {chain_id1} ---")
    count = counter.increment(chain_id1)
    print(f"Increment: Count = {count}")
    
    # すべてのカウンターの状態を表示
    # Display the state of all counters
    print("\n--- すべてのカウンターの状態 / State of All Counters ---")
    all_counts = counter.get_all_counts()
    for chain_id, count in all_counts.items():
        print(f"Chain ID: {chain_id}, Count: {count}")


if __name__ == "__main__":
    main()
