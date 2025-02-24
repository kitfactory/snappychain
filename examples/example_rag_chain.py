"""
Example of using RAG with document loading and querying.
RAGを使用したドキュメントの読み込みと質問応答の例。
"""

from snappychain import directory_load, build_rag_chain, recursive_split_text

def main():
    # Initialize data dictionary
    data = {
        "_dev": True,  
        "_session": {
            "documents": []
        }
    }

    # Load and split documents from the trees directory
    print("\nLoading and splitting documents...")
    print("ドキュメントを読み込んで分割中...")
    loader_chain = directory_load(
        directory_path="examples/documents/trees",
        show_progress=True
    ) | recursive_split_text(
        chunk_size=1000,
        chunk_overlap=200
    )

    result = loader_chain.invoke(data)
    documents = result["_session"]["documents"]

    # Configure RAG
    config = {
        "embeddings": {
            "provider": "openai",
            "model": "text-embedding-ada-002"
        },
        "vector_store": {
            "provider": "FAISS",
            "settings": {
                "persist_dir": "examples/vectorstore/rag"
            }
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini"
        }
    }

    # Build RAG and store documents
    try:
        print("\nBuilding RAG and storing documents...")
        print("RAGを構築してドキュメントを保存中...")
        rag = build_rag_chain(config)
        rag.store_documents(documents)
        
        # Sample questions
        questions = [
            "日本の木の特徴を教えてください",
            "これらの木の中で、最も大きくなる木は何ですか？",
            "これらの木の用途について教えてください"
        ]

        print("\nQuerying RAG with sample questions:")
        print("サンプルの質問でRAGに問い合わせ：")
        for i, question in enumerate(questions, 1):
            print(f"\n{i}. Question: {question}")
            try:
                answer = rag.query(question)
                print(f"Answer: {answer}")
            except Exception as e:
                print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
