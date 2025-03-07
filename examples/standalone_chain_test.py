"""
スタンドアロンのチェインIDテスト
Standalone chain ID test
"""

from langchain_core.runnables import RunnableLambda
import uuid


# チェインIDを管理するグローバル辞書
# Global dictionary to manage chain IDs
_chain_registry = {}


# チェインIDを生成する関数
# Function to generate chain ID
def generate_chain_id():
    """
    新しいチェインIDを生成する
    Generate a new chain ID
    """
    return str(uuid.uuid4())[:8]


# チェインにIDを設定する関数
# Function to set ID on a chain
def set_chain_id(chain, chain_id=None):
    """
    チェインにIDを設定する
    Set ID on a chain
    """
    chain_id = chain_id or generate_chain_id()
    _chain_registry[id(chain)] = chain_id
    return chain_id


# チェインからIDを取得する関数
# Function to get ID from a chain
def get_chain_id(chain):
    """
    チェインからIDを取得する
    Get ID from a chain
    """
    return _chain_registry.get(id(chain), "unknown")


# RunnableLambdaを取得する関数
# Function to get RunnableLambda
def create_runnable(func, chain_id=None):
    """
    関数をRunnableLambdaでラップして返す
    Wrap a function with RunnableLambda and return it
    """
    # 関数をRunnableLambdaでラップ
    # Wrap function with RunnableLambda
    runnable = RunnableLambda(func)
    
    # チェインIDを設定
    # Set chain ID
    set_chain_id(runnable, chain_id)
    
    return runnable


# パイプ演算子のオーバーライド関数
# Override function for pipe operator
def _override_pipe_operator():
    """
    パイプ演算子をオーバーライドする関数
    Function to override pipe operator
    """
    # 元のパイプ演算子を保存
    # Save original pipe operator
    original_pipe = RunnableLambda.__or__
    
    # パイプ演算子をオーバーライド
    # Override pipe operator
    def pipe_override(self, other):
        """
        パイプ演算子（|）をオーバーロードして、チェインIDを伝播する
        Overload pipe operator (|) to propagate chain ID
        """
        # 新しいチェインを作成
        # Create a new chain
        chain = original_pipe(self, other)
        
        # 左側のオペランドのIDを取得
        # Get ID from the left operand
        chain_id = get_chain_id(self)
        
        # IDが「unknown」でなければ、新しいチェインに設定
        # If ID is not "unknown", set it on the new chain
        if chain_id != "unknown":
            set_chain_id(chain, chain_id)
        
        return chain
    
    # パイプ演算子をオーバーライド
    # Override pipe operator
    RunnableLambda.__or__ = pipe_override


# テスト用の関数
# Test functions
def foo_function(x):
    """
    foo関数
    foo function
    """
    result = f"foo関数が実行されました: {x}"
    print(result)
    return result


def bar_function(x):
    """
    bar関数
    bar function
    """
    result = f"bar関数が実行されました: {x}"
    print(result)
    return result


def baz_function(x):
    """
    baz関数
    baz function
    """
    result = f"baz関数が実行されました: {x}"
    print(result)
    return result


def main():
    """
    メイン関数
    Main function
    """
    # パイプ演算子をオーバーライド
    # Override pipe operator
    _override_pipe_operator()
    
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
