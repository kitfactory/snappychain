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
        "rerank":"llm", # llm / 
        "provider": "openai",
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


# ログの塗分け-Verbose
print.py

デバッグ出力
　通常出力：指定なし

Verbose出力
　LLMに入力する最終的なプロンプト文字列：YELLOW
　LLMからの返答や出力：GREEN
　Rerank：MAZENTA
　VectorSearchなど各Retriver結果：CYAN
　エラー：赤色


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


# LangChain prebuild agent

* KnowledgeAgent:
    * Knowledge
    * RAGChain
    * LLMChain

* Knowledge
    * add_knowledge()
    * search_knowledge()
    * memory()


# 関数一覧

set_verbose(True) やchain.invoke(xxxx, verbose=True)で出力を詳細にするとのこと。

lambda で*args, **kwargsを受け取る必要があったり、verboseを確認する必要がある。

初期については
