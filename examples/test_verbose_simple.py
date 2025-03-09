"""
verboseパラメータの簡易テスト
Simple test for verbose parameter
"""

from snappychain import output
from snappychain.registry import ComponentRegistry
from snappychain.print import set_verbose

# verboseモードを有効化
# Enable verbose mode
set_verbose(True)

# レジストリのverboseを有効化
# Enable verbose for registry
registry = ComponentRegistry()
registry.set_verbose(True)

# メイン関数
# Main function
def main():
    # 出力パーサーを作成
    # Create output parser
    parser = output()
    
    # テストデータ
    # Test data
    test_data = {
        "_session": {
            "response": "これはテスト応答です。This is a test response."
        }
    }
    
    # verboseなしの呼び出し
    # Call without verbose
    print("verboseなしの呼び出し / Call without verbose:")
    result1 = parser.invoke(test_data)
    print(f"結果 / Result: {result1}")
    print("-" * 50)
    
    # verboseありの呼び出し
    # Call with verbose
    print("\nverboseありの呼び出し / Call with verbose:")
    result2 = parser.invoke(test_data, verbose=True)
    print(f"結果 / Result: {result2}")

if __name__ == "__main__":
    main()
