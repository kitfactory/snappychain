"""
LangChain互換のverboseおよびdebug出力機能を提供するモジュール
Module providing LangChain compatible verbose and debug output functionality
"""

from langchain.globals import get_verbose, get_debug, set_verbose as langchain_set_verbose
import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# 色のコード定義
# Color code definitions
class Color(Enum):
    """
    出力時の色を定義する列挙型
    Enumeration defining colors for output
    """
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

def set_verbose(verbose: bool) -> None:
    """
    verboseモードを設定する
    Set verbose mode
    
    Args:
        verbose (bool): verboseモードを有効にするかどうか / Whether to enable verbose mode
    """
    langchain_set_verbose(verbose)

def is_verbose() -> bool:
    """
    現在のverbose設定を取得する
    Get current verbose setting
    
    Returns:
        bool: verboseモードが有効かどうか / Whether verbose mode is enabled
    """
    return get_verbose()

def is_debug() -> bool:
    """
    現在のdebug設定を取得する
    Get current debug setting
    
    Returns:
        bool: debugモードが有効かどうか / Whether debug mode is enabled
    """
    return get_debug()

def _format_output(
    prefix: str, 
    data: Any, 
    color: Color,
    separator_length: int = 20
) -> None:
    """
    出力を整形する内部関数
    Internal function to format output
    
    Args:
        prefix (str): 出力の接頭辞 / Output prefix
        data (Any): 出力するデータ / Data to output
        color (Color): 出力の色 / Output color
        separator_length (int): 区切り線の長さ / Length of separator line
    """
    separator = f"\n{color.value}{'='*separator_length} {prefix} {'='*separator_length}{Color.RESET.value}"
    print(separator)
    
    if isinstance(data, (dict, list)):
        try:
            json_data = json.dumps(data, ensure_ascii=False, indent=2)
            print(f"{color.value}{json_data}{Color.RESET.value}")
        except:
            print(f"{color.value}{data}{Color.RESET.value}")
    else:
        print(f"{color.value}{data}{Color.RESET.value}")
    
    end_separator = f"{color.value}{'='*(separator_length*2 + len(prefix) + 2)}{Color.RESET.value}\n"
    print(end_separator)

def debug_print(
    prefix: str, 
    data: Any, 
    color: Color = Color.BLUE,
    separator_length: int = 20
) -> None:
    """
    デバッグ情報を出力する関数（LangChainのdebug設定に基づく）
    Function to output debug information (based on LangChain's debug setting)
    
    Args:
        prefix (str): 出力の接頭辞 / Output prefix
        data (Any): 出力するデータ / Data to output
        color (Color): 出力の色 / Output color
        separator_length (int): 区切り線の長さ / Length of separator line
    """
    if not get_debug():
        return
    
    _format_output(f"DEBUG: {prefix}", data, color, separator_length)

def verbose_print(
    prefix: str, 
    data: Any, 
    color: Color = Color.GREEN,
    separator_length: int = 20
) -> None:
    """
    詳細情報を出力する関数（LangChainのverbose設定に基づく）
    Function to output verbose information (based on LangChain's verbose setting)
    
    Args:
        prefix (str): 出力の接頭辞 / Output prefix
        data (Any): 出力するデータ / Data to output
        color (Color): 出力の色 / Output color
        separator_length (int): 区切り線の長さ / Length of separator line
    """
    if not get_verbose():
        return
    
    _format_output(prefix, data, color, separator_length)

def debug_request(data: Any) -> None:
    """
    リクエストデータをデバッグ出力する
    Debug output request data
    
    Args:
        data (Any): リクエストデータ / Request data
    """
    debug_print("リクエスト / Request", data, Color.GREEN)

def debug_response(data: Any) -> None:
    """
    レスポンスデータをデバッグ出力する
    Debug output response data
    
    Args:
        data (Any): レスポンスデータ / Response data
    """
    debug_print("レスポンス / Response", data, Color.YELLOW)

def debug_error(data: Any) -> None:
    """
    エラー情報をデバッグ出力する
    Debug output error information
    
    Args:
        data (Any): エラー情報 / Error information
    """
    debug_print("エラー / Error", data, Color.RED)

def debug_info(prefix: str, data: Any) -> None:
    """
    一般的な情報をデバッグ出力する
    Debug output general information
    
    Args:
        prefix (str): 出力の接頭辞 / Output prefix
        data (Any): 出力するデータ / Data to output
    """
    debug_print(prefix, data, Color.CYAN)
