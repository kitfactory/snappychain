"""
Example of using directory_load to load documents from a directory.
ディレクトリからドキュメントを読み込むdirectory_loadの使用例。
"""

from snappychain import directory_load, set_debug

def main():
    # デバッグモードを有効化
    # Enable debug mode
    set_debug(True)
    
    # Initialize data dictionary
    # データ辞書を初期化
    data = {
        "_session": {
            "documents": []
        }
    }

    # Load all documents from the trees directory
    # treesディレクトリから全てのドキュメントを読み込み
    loader = directory_load(
        directory_path="examples/documents/trees",
        show_progress=True,  # Show progress bar / 進捗バーを表示
        use_multithreading=True  # Enable multithreading / マルチスレッドを有効化
    )

    # Load the documents
    # ドキュメントを読み込み
    result = loader.invoke(data, verbose=True)

    # Print information about loaded documents
    # 読み込んだドキュメントの情報を表示
    documents = result["_session"]["documents"]
    print(f"\nLoaded {len(documents)} documents:")
    for i, doc in enumerate(documents, 1):
        print(f"\n{i}. Document from: {doc.metadata.get('source', 'Unknown source')}")
        print(f"   Content preview: {doc.page_content[:100]}...")

if __name__ == "__main__":
    main()
