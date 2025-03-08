"""
チェインのステップインデックスを使用する例
Example of using chain step indices
"""

import sys
import os
from typing import Dict, Any, List

# snappychainパッケージからチェイン関連の機能をインポート
# Import chain related functionality from snappychain package
from snappychain.chain import generate_chain_id, get_chain_id, set_chain_id, set_step_index, get_step_index


class ChainStepTracker:
    """
    チェインのステップを追跡するクラス
    Class to track chain steps
    """
    def __init__(self):
        """
        初期化
        Initialization
        """
        # チェインIDとステップ情報を保持する辞書
        # Dictionary to hold chain ID and step information
        self.chain_steps: Dict[str, List[Dict[str, Any]]] = {}
        
    def register_step(self, chain_id: str, step_name: str, step_index: int) -> None:
        """
        チェインのステップを登録する
        Register a chain step
        
        Args:
            chain_id (str): チェインID / Chain ID
            step_name (str): ステップ名 / Step name
            step_index (int): ステップインデックス / Step index
        """
        if chain_id not in self.chain_steps:
            self.chain_steps[chain_id] = []
        
        self.chain_steps[chain_id].append({
            "name": step_name,
            "index": step_index
        })
    
    def get_steps(self, chain_id: str) -> List[Dict[str, Any]]:
        """
        チェインのステップ情報を取得する
        Get step information for a chain
        
        Args:
            chain_id (str): チェインID / Chain ID
            
        Returns:
            List[Dict[str, Any]]: ステップ情報のリスト / List of step information
        """
        return self.chain_steps.get(chain_id, [])
    
    def get_all_chains(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        すべてのチェインのステップ情報を取得する
        Get step information for all chains
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: チェインIDとステップ情報のリスト / Dictionary of chain IDs and step information lists
        """
        return self.chain_steps.copy()


# グローバルなステップトラッカーインスタンス
# Global step tracker instance
tracker = ChainStepTracker()


def step1(input_data: Any, chain_id: str = None, step_index: int = 0) -> Dict[str, Any]:
    """
    チェインの最初のステップ
    First step of the chain
    
    Args:
        input_data (Any): 入力データ / Input data
        chain_id (str, optional): チェインID / Chain ID
        step_index (int, optional): ステップインデックス / Step index
        
    Returns:
        Dict[str, Any]: 処理結果 / Processing result
    """
    # ステップ情報を登録
    # Register step information
    tracker.register_step(chain_id, "step1", step_index)
    
    print(f"Step 1 executed (Chain ID: {chain_id}, Step Index: {step_index})")
    return {"input": input_data, "step": 1, "chain_id": chain_id, "step_index": step_index}


def step2(input_data: Dict[str, Any], chain_id: str = None, step_index: int = 1) -> Dict[str, Any]:
    """
    チェインの2番目のステップ
    Second step of the chain
    
    Args:
        input_data (Dict[str, Any]): 前のステップからの入力 / Input from the previous step
        chain_id (str, optional): チェインID / Chain ID
        step_index (int, optional): ステップインデックス / Step index
        
    Returns:
        Dict[str, Any]: 処理結果 / Processing result
    """
    # 前のステップからチェインIDを取得
    # Get chain ID from the previous step
    chain_id = input_data.get("chain_id", chain_id)
    
    # ステップ情報を登録
    # Register step information
    tracker.register_step(chain_id, "step2", step_index)
    
    print(f"Step 2 executed (Chain ID: {chain_id}, Step Index: {step_index})")
    return {**input_data, "step": 2, "step_index": step_index}


def step3(input_data: Dict[str, Any], chain_id: str = None, step_index: int = 2) -> Dict[str, Any]:
    """
    チェインの3番目のステップ
    Third step of the chain
    
    Args:
        input_data (Dict[str, Any]): 前のステップからの入力 / Input from the previous step
        chain_id (str, optional): チェインID / Chain ID
        step_index (int, optional): ステップインデックス / Step index
        
    Returns:
        Dict[str, Any]: 処理結果 / Processing result
    """
    # 前のステップからチェインIDを取得
    # Get chain ID from the previous step
    chain_id = input_data.get("chain_id", chain_id)
    
    # ステップ情報を登録
    # Register step information
    tracker.register_step(chain_id, "step3", step_index)
    
    print(f"Step 3 executed (Chain ID: {chain_id}, Step Index: {step_index})")
    return {**input_data, "step": 3, "step_index": step_index}


def execute_chain(input_data: Any, chain_id: str = None) -> Dict[str, Any]:
    """
    チェインを実行する
    Execute a chain
    
    Args:
        input_data (Any): 入力データ / Input data
        chain_id (str, optional): チェインID / Chain ID
        
    Returns:
        Dict[str, Any]: 処理結果 / Processing result
    """
    # チェインIDが指定されていない場合は生成
    # Generate chain ID if not specified
    if chain_id is None:
        chain_id = generate_chain_id()
    
    # ステップを順番に実行
    # Execute steps in order
    result = step1(input_data, chain_id, 0)
    result = step2(result, chain_id, 1)
    result = step3(result, chain_id, 2)
    
    return result


def main():
    """
    メイン関数
    Main function
    """
    print("=== チェインステップインデックスの例 / Chain Step Index Example ===")
    
    # 最初のチェインを実行
    # Execute the first chain
    print("\n--- 最初のチェイン実行 / First Chain Execution ---")
    chain_id1 = generate_chain_id()
    result1 = execute_chain("Hello, Chain!", chain_id1)
    print(f"Result: {result1}")
    
    # 新しいチェインを実行
    # Execute a new chain
    print("\n--- 新しいチェイン実行 / New Chain Execution ---")
    chain_id2 = generate_chain_id()
    result2 = execute_chain("Hello, New Chain!", chain_id2)
    print(f"Result: {result2}")
    
    # すべてのチェインのステップ情報を表示
    # Display step information for all chains
    print("\n--- すべてのチェインのステップ情報 / Step Information for All Chains ---")
    all_chains = tracker.get_all_chains()
    for chain_id, steps in all_chains.items():
        print(f"\nChain ID: {chain_id}")
        for step in steps:
            print(f"  Step Name: {step['name']}, Step Index: {step['index']}")


if __name__ == "__main__":
    main()
