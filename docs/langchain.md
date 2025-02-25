# LangChainの考え方

## プロンプト
・PromptTemplate : 全指定
・ChatPromptTemplate : Message型が指定が必要/ system・human・ai
　Templateにはinvoke()メソッドが存在。
・MessagesPlaceHolder: 履歴を記録

# OutputParser
・ llmの後で出力の取り出し方を決める
・ PydanticOutputParser: Pydanticのデータ・オブジェクトになるように出力
・ StrOutputParser: 文字列になるように出力
・ JsonOutputParser: JSONを取り出しに使用する。

・ get_format_instructions(): 出力の形式を指定するプロンプトを得る

# ResponseSchema
・ ResponseSchema: 出力の形式を指定する
・ OutputParserでget_format_instructions()に変換

まずはここまでの完成

# RAGに必要な要素

* DocumentLoader
    * 文書を読み出し。拡張子に応じて。

* DocumentSplitter
* Embeddings
* VectorStore


# RAG
* RetrivalQAがあるが、準備が必要
* 


* ただ、RagChain.run()

```
config = {
    "embeddings": {
        "provider":"openai",
        "model":"text-embedding-ada-002"
    }
    "vector_store": {
        "provider":"FAISS",
        "settings":{
            "persist_dir":"dir_of_persist"
        }
    },
    "llm":{
        "provider":"openai",
        "model":"gpt-4o"
    },
    "bm25":{
        "settings":{
            "persist_dir":"bm25"
        },
    },
    "rerank":{
        "provider":"llm", # llm / cohere
        "model":"gpt-4o",
    },
}

rag.pyでの提供メソッドは以下を提供する。

・build_rag_chain(config)
・rag_chain.store_documents()
・rag_chain.query()


```サンプル
documents = directory_load().to_documents()
config = {
    "embeddings": {
        "provider":"openai",
        "model":"text-embedding-ada-002"
    }
    "vector_store": {
        "provider":"FAISS",
        "settings":{
            "persist_dir":"dir_of_persist"
        }
    },
    "llm":{
        "provider":"openai",
        "model":"gpt-4o"
    },
}

rag = build_rag_chain(config)
rag = rag.store_documents(documents)
rag.query("query")
```




# LangGraph

基本的にはGraphに以下の操作を行う
・Node追加：ノード
・Edge追加：ノードの間を繋ぐ

・conditional_edge : 条件判定
・ToolNode:外部を呼び出すノード

SnappyGraph.node(").conditional_edge({
    "condition":"state",
})


SnappyGraph({
    nodes:
        tool_node(),
    edges:
        conditonal_edge(),
})

# ログの塗分け

・通常出力：指定なし（現在色）
・エラー：赤色
・リクエスト・入力：緑色
・返答や出力：橙色
・Rerank結果：青
・VectorStoreサーチ結果：青
・検証結果：正常　青/異常赤（これはまだない、今後）