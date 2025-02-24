from langchain_openai import OpenAIEmbeddings
from snappychain import (
    directory_load,
    recursive_split_text,
    get_chain_documents,
    UnifiedVectorStore
)

# FAISSの場合の設定
faiss_settings = {
    "provider": "faiss",
    "save_dir": "examples/vectorstores/faiss",
}

# Chromaの場合の設定
chroma_settings = {
    "provider": "chroma",
    "save_dir": "examples/vectorstores/chroma",
}

if __name__ == "__main__":
    import shutil
    import os

    # FAISSのテスト
    print("\nTesting FAISS vector store...")
    print("--------------------------------")
    
    directory = "examples/vectorstores/faiss"
    if os.path.exists(directory):
        print("Removing", directory)
        shutil.rmtree(directory)

    uv_faiss = UnifiedVectorStore(
        settings=faiss_settings,
        embeddings=OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )        
    )

    # Chromaのテスト
    print("\nTesting Chroma vector store...")
    print("--------------------------------")
    
    directory = "examples/vectorstores/chroma"
    if os.path.exists(directory):
        print("Removing", directory)
        shutil.rmtree(directory)

    uv_chroma = UnifiedVectorStore(
        settings=chroma_settings,
        embeddings=OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )        
    )

    # ドキュメントの読み込みと分割
    loader_chain = directory_load(
        directory_path="examples/documents/trees",
        show_progress=True
    ) | recursive_split_text(
        chunk_size=1000,
        chunk_overlap=200
    ) | get_chain_documents()

    documents = loader_chain.invoke({})
    print("\nDocuments loaded:", len(documents))

    # FAISSへの保存とテスト
    print("\nTesting FAISS storage and search...")
    uv_faiss.add_documents(documents)
    result = uv_faiss.similarity_search("ヤマボウシはいつ頃咲きますか？", k=3)
    print("FAISS Results:", len(result))
    for r in result:
        print(r)

    # Chromaへの保存とテスト
    print("\nTesting Chroma storage and search...")
    uv_chroma.add_documents(documents)
    result = uv_chroma.similarity_search("アオダモの花は何色？", k=1)
    print("Chroma Results:", len(result))
    for r in result:
        print(r)
