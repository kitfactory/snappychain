from snappychain.loader import text_load, pypdf_load  # Import text_load function from snappychain.loader // snappychain.loaderからtext_load関数をインポートします

def main_text_load():
    """
    Main function to demonstrate the usage of text_load.
    text_loadの使用例を示すメイン関数です。
    """
    # Define the list of file paths to load text documents // テキストドキュメントを読み込むためのファイルパスのリストを定義します
    file_paths = [
        "examples/documents/text_a.txt",  # Path to text_a.txt // text_a.txtへのパス
        "examples/documents/text_b.txt",  # Path to text_b.txt // text_b.txtへのパス
        "examples/documents/text_c.txt"   # Path to text_c.txt // text_c.txtへのパス
    ]
    
    # Obtain a RunnableLambda from text_load function // text_load関数からRunnableLambdaを取得します
    runnable = text_load(file_paths)
    
    # Initialize empty data dictionary // 空のデータ辞書を初期化します
    data = {}
    
    # Invoke the runnable to load documents and update data // Runnableを実行してドキュメントを読み込み、データを更新します
    result = runnable.invoke(data)
    
    # Retrieve the loaded documents from the session // セッションから読み込んだドキュメントを取得します
    documents = result.get("_session", {}).get("documents", [])
    
    # Print the loaded documents // 読み込んだドキュメントを表示します
    print("Loaded Documents:")  # English: Loaded Documents // 日本語: 読み込んだドキュメント
    for doc in documents:
        print(doc)


# Example of loading PDF documents from the 'documetns' directory
# 例: 'documetns' ディレクトリからPDFドキュメントを読み込む例
import os


def main_pypdf_load():
    # Define the directory containing the PDF files
    # PDFファイルが存在するディレクトリを定義します
    documents_dir = os.path.join(os.path.dirname(__file__), "documents")

    # Retrieve all PDF file paths in the directory
    # ディレクトリ内のすべてのPDFファイルパスを取得します
    pdf_files = [os.path.join(documents_dir, f) for f in os.listdir(documents_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {documents_dir}")
        return

    # Create an initial data dictionary
    # 初期データ辞書を作成します
    data = {}

    # Load the PDF documents using pypdf_load
    # pypdf_loadを使用してPDFドキュメントを読み込みます
    result = pypdf_load(pdf_files).invoke(data)

    # Extract the loaded documents from the session
    # セッションから読み込んだドキュメントを抽出します
    documents = result.get("_session", {}).get("documents", [])

    print("Loaded documents:")
    for doc in documents:
        print(doc)


if __name__ == "__main__":
    main_text_load()  # Run the main function // メイン関数を実行します
    main_pypdf_load()
