"""
チェインID管理と基本的なチェイン機能を提供するモジュール
Module providing chain ID management and basic chain functionality
"""

from langchain_core.runnables import RunnableLambda, Runnable
import uuid
from typing import Dict, List, Any, Optional, Union, Callable

# チェインIDを管理するグローバル辞書
# Global dictionary to manage chain IDs
_chain_registry = {}

# チェインのステップインデックスを管理するグローバル辞書
# Global dictionary to manage chain step indices
_chain_step_indices = {}

# チェインの現在のステップ数を管理するグローバル辞書
# Global dictionary to manage current step count of chains
_chain_step_counts = {}


def generate_chain_id():
    """
    新しいチェインIDを生成する
    Generate a new chain ID
    
    Returns:
        str: チェインID / Chain ID
    """
    return str(uuid.uuid4()).replace('-', '')[:16]


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
    chain_id = chain_id or generate_chain_id()
    _chain_registry[id(chain)] = chain_id
    
    # チェインのステップカウントを初期化
    # Initialize step count for the chain
    if chain_id not in _chain_step_counts:
        _chain_step_counts[chain_id] = 0
        
    return chain_id


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


def set_step_index(chain, index=None):
    """
    チェインのステップにインデックスを設定する
    Set index on a chain step
    
    Args:
        chain: チェインステップ / Chain step
        index (int, optional): ステップインデックス / Step index
    
    Returns:
        int: ステップインデックス / Step index
    """
    chain_id = get_chain_id(chain)
    
    if index is None:
        # チェインのステップカウントをインクリメント
        # Increment step count for the chain
        if chain_id in _chain_step_counts:
            _chain_step_counts[chain_id] += 1
            index = _chain_step_counts[chain_id]
        else:
            index = 0
    
    _chain_step_indices[id(chain)] = index
    return index


def get_step_index(chain):
    """
    チェインのステップからインデックスを取得する
    Get index from a chain step
    
    Args:
        chain: チェインステップ / Chain step
    
    Returns:
        int: ステップインデックス / Step index
    """
    return _chain_step_indices.get(id(chain), 0)


def create_runnable(func, chain_id=None, step_index=None):
    """
    関数をRunnableLambdaでラップして返す
    Wrap a function with RunnableLambda and return it
    
    Args:
        func: ラップする関数 / Function to wrap
        chain_id (str, optional): チェインID / Chain ID
        step_index (int, optional): ステップインデックス / Step index
    
    Returns:
        RunnableLambda: ラップされた関数 / Wrapped function
    """
    runnable = RunnableLambda(func)
    chain_id = set_chain_id(runnable, chain_id)
    set_step_index(runnable, step_index)
    return runnable


class Chain(RunnableLambda):
    """
    チェインクラス
    Chain class
    """
    def __init__(self, func, chain_id=None, step_index=None):
        """
        初期化
        Initialization
        
        Args:
            func: ラップする関数 / Function to wrap
            chain_id (str, optional): チェインID / Chain ID
            step_index (int, optional): ステップインデックス / Step index
        """
        super().__init__(func)
        self.chain_id = chain_id or generate_chain_id()
        _chain_registry[id(self)] = self.chain_id
        
        # ステップインデックスを設定
        # Set step index
        self.step_index = step_index
        if step_index is None:
            # チェインのステップカウントをインクリメント
            # Increment step count for the chain
            if self.chain_id in _chain_step_counts:
                _chain_step_counts[self.chain_id] += 1
                self.step_index = _chain_step_counts[self.chain_id]
            else:
                _chain_step_counts[self.chain_id] = 0
                self.step_index = 0
        
        _chain_step_indices[id(self)] = self.step_index

    def __or__(self, other):
        """
        パイプ演算子（|）をオーバーロード
        Overload pipe operator (|)
        
        Args:
            other: 右側のオペランド / Right operand
        
        Returns:
            Runnable: 新しいチェイン / New chain
        """
        result = super().__or__(other)
        set_chain_id(result, self.chain_id)
        return result

    def __repr__(self):
        """
        文字列表現
        String representation
        
        Returns:
            str: 文字列表現 / String representation
        """
        return f"Chain(chain_id={self.chain_id}, step_index={self.step_index})"

    def invoke(self, input, *args, **kwargs):
        """
        チェインを実行する
        Execute the chain
        
        Args:
            input: 入力データ / Input data
            *args: 可変長位置引数 / Variable length positional arguments
            **kwargs: キーワード引数 / Keyword arguments
            
        Returns:
            Any: チェインの実行結果 / Chain execution result
        """
        # 入力がNoneの場合は空の辞書を使用
        # Use empty dictionary if input is None
        if input is None:
            input = {}
            
        # 入力が辞書でない場合は、辞書に変換
        # Convert input to dictionary if it is not a dictionary
        if not isinstance(input, dict):
            input = {"input": input}
            
        # セッションを初期化
        # Initialize session
        if "_session" not in input:
            input["_session"] = {}
            
        # チェインIDとステップインデックスをセッションに保存
        # Save chain ID and step index to session
        session = input["_session"]
        session["chain"] = self
        
        # argsとkwargsをセッションに保存
        # Save args and kwargs to session
        if args:
            session["args"] = args
            
        # 既存のkwargsと新しいkwargsをマージ
        # Merge existing kwargs with new kwargs
        if "kwargs" not in session:
            session["kwargs"] = {}
            
        if not isinstance(session["kwargs"], dict):
            session["kwargs"] = {}
            
        session["kwargs"].update(kwargs)
        
        # 親クラスのinvokeメソッドを呼び出す
        # Call the invoke method of the parent class
        try:
            return super().invoke(input)
        except Exception as e:
            print(f"チェインの実行中にエラーが発生しました / Error occurred during chain execution: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def chain(func=None, chain_id=None, step_index=None):
    """
    関数をChainでラップして返す
    Wrap a function with Chain and return it
    
    Args:
        func: ラップする関数 / Function to wrap
        chain_id (str, optional): チェインID / Chain ID
        step_index (int, optional): ステップインデックス / Step index
    
    Returns:
        Chain: ラップされた関数 / Wrapped function
    """
    if func is None:
        # デコレータとして使用する場合
        # When used as a decorator
        def decorator(f):
            return Chain(f, chain_id, step_index)
        return decorator
    return Chain(func, chain_id, step_index)
