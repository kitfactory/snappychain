# -*- coding: utf-8 -*-
"""
Example script for testing splitter functions in SnappyChain.
スプリッター関数の動作を確認するためのサンプルスクリプトです。
"""

from snappychain.splitter import markdown_text_splitter, json_text_splitter, python_text_splitter
from langchain.schema import Document
import traceback
import sys


def load_file_content(file_path: str) -> str:
    """
    Load content from a file.
    ファイルから内容を読み込みます。

    Args:
        file_path (str): Path to the file to load.
                        読み込むファイルのパス。

    Returns:
        str: Content of the file.
             ファイルの内容。
    """
    try:
        print(f"Loading file: {file_path}", file=sys.stderr)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"File content length: {len(content)} characters", file=sys.stderr)
        return content
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise


def test_splitter(splitter_func, splitter_name: str, file_path: str):
    """
    Test a given splitter function and print the results.
    指定されたスプリッター関数をテストし、結果を表示します。

    Args:
        splitter_func: A no-argument function returning a RunnableLambda that processes documents.
                      ドキュメントを処理するRunnableLambdaを返す引数なしの関数。
        splitter_name (str): Name of the splitter for display purposes.
                            表示用のスプリッター名。
        file_path (str): Path to the file to test with.
                        テスト用のファイルパス。
    """
    try:
        print(f"\n--- Testing {splitter_name} with {file_path} ---", file=sys.stderr)
        
        # Load and create a sample Document
        # サンプルDocumentを作成します
        content = load_file_content(file_path)
        print(f"Creating Document with content length: {len(content)}", file=sys.stderr)
        sample_doc = Document(page_content=content, metadata={"source": file_path})
        
        # Initialize data with the sample Document
        # サンプルDocumentでデータを初期化します
        data = {"_session": {"documents": [sample_doc]}}
        
        # Invoke the splitter function
        # スプリッター関数を実行します
        print(f"Invoking {splitter_name}", file=sys.stderr)
        result = splitter_func().invoke(data)
        
        # Retrieve and print the resulting Document objects
        # 結果として得られたDocumentオブジェクトの内容を表示します
        docs = result.get("_session", {}).get("documents", [])
        print(f"\nResults from {splitter_name}:", file=sys.stderr)
        for idx, doc in enumerate(docs, start=1):
            print(f"\nDocument {idx}:", file=sys.stderr)
            print("-" * 40, file=sys.stderr)
            print(doc.page_content, file=sys.stderr)
            print("-" * 40, file=sys.stderr)
    except Exception as e:
        print(f"Error in {splitter_name}: {str(e)}", file=sys.stderr)
        print("Traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


def main():
    """
    Main function to test all splitter functions with appropriate sample files.
    全てのスプリッター関数を適切なサンプルファイルでテストするメイン関数です。
    """
    try:
        # Test markdown splitter with text file
        # テキストファイルでMarkdownスプリッターをテスト
        test_splitter(
            markdown_text_splitter,
            "MarkdownTextSplitter",
            "examples/documents/sample.txt"
        )
        
        # Test JSON splitter with JSON file
        # JSONファイルでJSONスプリッターをテスト
        test_splitter(
            json_text_splitter,
            "JSONTextSplitter",
            "examples/documents/sample.json"
        )
        
        # Test Python splitter with Python file
        # PythonファイルでPythonスプリッターをテスト
        test_splitter(
            python_text_splitter,
            "PythonTextSplitter",
            "examples/documents/sample.py"
        )
    except Exception as e:
        print(f"Error in main: {str(e)}", file=sys.stderr)
        print("Traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


if __name__ == '__main__':
    main()
