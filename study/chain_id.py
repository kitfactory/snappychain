"""
チェイン作成時にチェインごとに一意のIDを持たせる最もシンプルな方法
The simplest way to assign a unique ID to each chain during chain creation
"""

from langchain_core.runnables import RunnableLambda
import uuid
import sys


# チェインIDを管理するグローバル辞書
# Global dictionary to manage chain IDs
_chain_registry = {}


# チェインIDを生成する関数
# Function to generate chain ID
def generate_chain_id():
    """
    新しいチェインIDを生成する
    Generate a new chain ID
    
    Returns:
        str: チェインID / Chain ID
    """
    return str(uuid.uuid4())[:8]  # 短いIDを使用 / Use a short ID


# チェインにIDを設定する関数
# Function to set ID on a chain
def set_chain_id(chain, chain_id=None):
    """
    チェインにIDを設定する
    Set ID on a chain
    
    Args:
        chain: チェイン / Chain
        chain_id (str, optional): チェインID / Chain ID
    
    Returns:
        str: チェインID / Chain ID
    """
    # チェインIDが指定されていない場合は新しいIDを生成
    # Generate a new ID if chain_id is not specified
    chain_id = chain_id or generate_chain_id()
    
    # チェインIDをレジストリに登録
    # Register chain ID in the registry
    _chain_registry[id(chain)] = chain_id
    
    return chain_id


# チェインからIDを取得する関数
# Function to get ID from a chain
def get_chain_id(chain):
    """
    チェインからIDを取得する
    Get ID from a chain
    
    Args:
        chain: チェイン / Chain
    
    Returns:
        str: チェインID / Chain ID
    """
    return _chain_registry.get(id(chain), "unknown")


# 元のRunnableLambdaコンストラクタを保存
# Save original RunnableLambda constructor
if not hasattr(RunnableLambda, '_original_init'):
    RunnableLambda._original_init = RunnableLambda.__init__
    
    # RunnableLambdaコンストラクタをオーバーライド
    # Override RunnableLambda constructor
    def _init_override(self, func, *args, chain_id=None, **kwargs):
        """
        RunnableLambdaコンストラクタをオーバーライドして、チェインIDを設定できるようにする
        Override RunnableLambda constructor to allow setting chain ID
        
        Args:
            self: RunnableLambdaインスタンス / RunnableLambda instance
            func: 関数 / Function
            chain_id (str, optional): チェインID / Chain ID
            *args, **kwargs: その他の引数 / Other arguments
        """
        # 元のコンストラクタを呼び出す
        # Call original constructor
        self._original_init(func, *args, **kwargs)
        
        # チェインIDを設定（指定されていない場合は新しいIDを生成）
        # Set chain ID (generate a new ID if not specified)
        set_chain_id(self, chain_id)
    
    # コンストラクタをオーバーライド
    # Override constructor
    RunnableLambda.__init__ = _init_override


# パイプ演算子のオーバーロード
# Overload pipe operator
def _pipe_override(self, other):
    """
    パイプ演算子（|）をオーバーロードして、チェインIDを伝播する
    Overload pipe operator (|) to propagate chain ID
    
    Args:
        self: 左側のオペランド / Left operand
        other: 右側のオペランド / Right operand
    
    Returns:
        チェイン / Chain
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


# RunnableLambdaのパイプ演算子をオーバーライド
# Override pipe operator of RunnableLambda
if not hasattr(RunnableLambda, '_original_pipe'):
    RunnableLambda._original_pipe = RunnableLambda.__or__
    RunnableLambda.__or__ = _pipe_override


# 実行ログを保存する変数
# Variable to store execution logs
execution_logs = []


# サンプルコンポーネント
# Sample components
def foo_chain(*args, **kwargs):
    """
    サンプルコンポーネント1
    Sample component 1
    """
    execution_logs.append("foo_chain が実行されました / foo_chain executed")
    return args, kwargs


def bar_chain(*args, **kwargs):
    """
    サンプルコンポーネント2
    Sample component 2
    """
    execution_logs.append("bar_chain が実行されました / bar_chain executed")
    return args, kwargs


def baz_chain(*args, **kwargs):
    """
    サンプルコンポーネント3
    Sample component 3
    """
    execution_logs.append("baz_chain が実行されました / baz_chain executed")
    return args, kwargs


# RunnableLambdaを取得する関数
# Functions to get RunnableLambda
def get_foo(chain_id=None):
    """
    foo_chainのRunnableLambdaを取得する
    Get RunnableLambda for foo_chain
    
    Args:
        chain_id (str, optional): チェインID / Chain ID
    
    Returns:
        RunnableLambda: foo_chainのRunnableLambda / RunnableLambda for foo_chain
    """
    return RunnableLambda(foo_chain, chain_id=chain_id)


def get_bar(chain_id=None):
    """
    bar_chainのRunnableLambdaを取得する
    Get RunnableLambda for bar_chain
    
    Args:
        chain_id (str, optional): チェインID / Chain ID
    
    Returns:
        RunnableLambda: bar_chainのRunnableLambda / RunnableLambda for bar_chain
    """
    return RunnableLambda(bar_chain, chain_id=chain_id)


def get_baz(chain_id=None):
    """
    baz_chainのRunnableLambdaを取得する
    Get RunnableLambda for baz_chain
    
    Args:
        chain_id (str, optional): チェインID / Chain ID
    
    Returns:
        RunnableLambda: baz_chainのRunnableLambda / RunnableLambda for baz_chain
    """
    return RunnableLambda(baz_chain, chain_id=chain_id)


# 実行例
# Example execution
if __name__ == "__main__":
    # 出力バッファ
    output = []
    
    output.append("\n" + "="*60)
    output.append("=== get_xxx関数を使用する例 / Example of using get_xxx functions ===")
    output.append("="*60)
    
    # 明示的にIDを指定
    # Specify ID explicitly
    explicit_id = generate_chain_id()
    output.append("\n" + "-"*60)
    output.append(f"明示的にIDを指定 / Specify ID explicitly: {explicit_id}")
    output.append("-"*60)
    
    chain1 = get_foo(chain_id=explicit_id) | get_bar()
    output.append(f"chain1.chain_id: {get_chain_id(chain1)}")
    
    # チェインを実行
    # Execute chain
    output.append("\nチェイン1を実行 / Execute chain 1:")
    chain1.invoke({})
    for log in execution_logs:
        output.append(log)
    execution_logs.clear()
    
    # 自動的にIDを生成
    # Generate ID automatically
    output.append("\n" + "-"*60)
    output.append("自動的にIDを生成 / Generate ID automatically:")
    output.append("-"*60)
    
    chain2 = get_foo() | get_baz()
    output.append(f"chain2.chain_id: {get_chain_id(chain2)}")
    
    # チェインを実行
    # Execute chain
    output.append("\nチェイン2を実行 / Execute chain 2:")
    chain2.invoke({})
    for log in execution_logs:
        output.append(log)
    execution_logs.clear()
    
    # 同じIDを持つチェインの例
    # Example of chains with the same ID
    output.append("\n" + "="*60)
    output.append("=== 同じIDを持つチェインの例 / Example of chains with the same ID ===")
    output.append("="*60)
    
    # 同じIDを持つコンポーネント
    # Components with the same ID
    shared_id = generate_chain_id()
    output.append("\n" + "-"*60)
    output.append(f"同じIDを共有 / Share the same ID: {shared_id}")
    output.append("-"*60)
    
    chain3 = get_foo(chain_id=shared_id) | get_bar()
    chain4 = get_foo(chain_id=shared_id) | get_baz()
    
    output.append(f"chain3.chain_id: {get_chain_id(chain3)}")
    output.append(f"chain4.chain_id: {get_chain_id(chain4)}")
    
    # チェインを実行
    # Execute chains
    output.append("\nチェイン3を実行 / Execute chain 3:")
    chain3.invoke({})
    for log in execution_logs:
        output.append(log)
    execution_logs.clear()
    
    output.append("\nチェイン4を実行 / Execute chain 4:")
    chain4.invoke({})
    for log in execution_logs:
        output.append(log)
    execution_logs.clear()
    
    output.append("\n" + "="*60)
    output.append("=== 終了 / End ===")
    output.append("="*60)
    
    # 結果をファイルに書き出す
    # Write results to file
    with open("chain_id_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))
    
    # コンソールにも出力
    # Also output to console
    print("実行結果をchain_id_results.txtに書き出しました。")
    print("Results have been written to chain_id_results.txt.")
