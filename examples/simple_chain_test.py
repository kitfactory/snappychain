"""
シンプルなチェインIDのテスト
Simple test for chain ID
"""

from langchain_core.runnables import RunnableLambda
import uuid


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
    # 元のパイプ演算子を保存
    # Save original pipe operator
    if not hasattr(RunnableLambda, '_original_pipe'):
        RunnableLambda._original_pipe = RunnableLambda.__or__
        
        # パイプ演算子をオーバーライド
        # Override pipe operator
        def _pipe_override(self, other):
            """
            パイプ演算子（|）をオーバーロードして、チェインIDを伝播する
            Overload pipe operator (|) to propagate chain ID
            """
            # 新しいチェインを作成
            # Create a new chain
            chain = self._original_pipe(other)
            
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
        RunnableLambda.__or__ = _pipe_override
    
    # 関数をRunnableLambdaでラップ
    # Wrap function with RunnableLambda
    runnable = RunnableLambda(func)
    
    # チェインIDを設定
    # Set chain ID
    set_chain_id(runnable, chain_id)
    
    return runnable


def main():
    """
    メイン関数
    Main function
    """
    print("=== チェインID機能のテスト / Test of chain ID functionality ===")
    
    # 明示的にIDを指定したチェインの作成
    # Create a chain with explicitly specified ID
    explicit_id = "c8ed7ce7"
    foo_runnable = create_runnable(foo_function, chain_id=explicit_id)
    bar_runnable = create_runnable(bar_function)
    
    # パイプ演算子でチェインを作成
    # Create a chain using pipe operator
    chain1 = foo_runnable | bar_runnable
    
    print(f"chain1.chain_id: {get_chain_id(chain1)}")
    print("チェイン1を実行 / Execute chain 1:")
    chain1.invoke("テスト入力 / Test input")
    
    print("テスト完了 / Test completed")


if __name__ == "__main__":
    main()
