"""
チェインIDの使用例
Example of using chain ID
"""

import sys
import os

# プロジェクトのルートディレクトリをパスに追加
# Add project root directory to path
# 
# snappychainパッケージからチェインID関連の機能をインポート
# Import chain ID related functions from snappychain package
from snappychain import (
    generate_chain_id,
    set_chain_id,
    get_chain_id,
    create_runnable,
    Chain,
    chain,
)

# テスト用の関数
# Test functions
def foo_function(x):
    """
    foo関数
    foo function
    
    Args:
        x: 入力 / Input
    
    Returns:
        str: 出力 / Output
    """
    result = f"foo関数が実行されました: {x}"
    print(result)
    return result


def bar_function(x):
    """
    bar関数
    bar function
    
    Args:
        x: 入力 / Input
    
    Returns:
        str: 出力 / Output
    """
    result = f"bar関数が実行されました: {x}"
    print(result)
    return result


def baz_function(x):
    """
    baz関数
    baz function
    
    Args:
        x: 入力 / Input
    
    Returns:
        str: 出力 / Output
    """
    result = f"baz関数が実行されました: {x}"
    print(result)
    return result


def main():
    """
    メイン関数
    Main function
    """
    print("============================================================")
    print("=== チェインID機能のテスト / Test of chain ID functionality ===")
    print("============================================================")
    print("")

    # 明示的にIDを指定したチェインの作成
    # Create a chain with explicitly specified ID
    print("------------------------------------------------------------")
    print("明示的にIDを指定 / Specify ID explicitly")
    print("------------------------------------------------------------")
    
    explicit_id = "c8ed7ce7"
    foo_runnable = create_runnable(foo_function, chain_id=explicit_id)
    bar_runnable = create_runnable(bar_function)
    
    # パイプ演算子でチェインを作成
    # Create a chain using pipe operator
    chain1 = foo_runnable | bar_runnable
    
    print(f"chain1.chain_id: {get_chain_id(chain1)}")
    print("")
    print("チェイン1を実行 / Execute chain 1:")
    chain1.invoke("テスト入力 / Test input")
    print("")

    # 自動的にIDを生成するチェインの作成
    # Create a chain with automatically generated ID
    print("------------------------------------------------------------")
    print("自動的にIDを生成 / Generate ID automatically")
    print("------------------------------------------------------------")
    
    foo_runnable2 = create_runnable(foo_function)
    baz_runnable = create_runnable(baz_function)
    
    # パイプ演算子でチェインを作成
    # Create a chain using pipe operator
    chain2 = foo_runnable2 | baz_runnable
    
    print(f"chain2.chain_id: {get_chain_id(chain2)}")
    print("")
    print("チェイン2を実行 / Execute chain 2:")
    chain2.invoke("別のテスト入力 / Another test input")
    print("")

    # 同じIDを持つ複数のチェインの作成
    # Create multiple chains with the same ID
    print("============================================================")
    print("=== 同じIDを持つチェインの例 / Example of chains with the same ID ===")
    print("============================================================")
    print("")
    
    print("------------------------------------------------------------")
    shared_id = "a622ea8b"
    print(f"同じIDを共有 / Share the same ID: {shared_id}")
    print("------------------------------------------------------------")
    
    foo_runnable3 = create_runnable(foo_function, chain_id=shared_id)
    bar_runnable3 = create_runnable(bar_function)
    baz_runnable3 = create_runnable(baz_function)
    
    # 同じIDを持つ2つのチェインを作成
    # Create two chains with the same ID
    chain3 = foo_runnable3 | bar_runnable3
    chain4 = foo_runnable3 | baz_runnable3
    
    print(f"chain3.chain_id: {get_chain_id(chain3)}")
    print(f"chain4.chain_id: {get_chain_id(chain4)}")
    print("")
    
    print("チェイン3を実行 / Execute chain 3:")
    chain3.invoke("共有IDテスト1 / Shared ID test 1")
    print("")
    
    print("チェイン4を実行 / Execute chain 4:")
    chain4.invoke("共有IDテスト2 / Shared ID test 2")
    print("")
    
    print("============================================================")
    print("=== 終了 / End ===")
    print("============================================================")


if __name__ == "__main__":
    main()
