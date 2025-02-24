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
* DocumentSplitter
* Embeddings
* VectorStore


* RetrivalQA



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
