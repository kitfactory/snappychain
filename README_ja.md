# SnappyChain 

SnappyChainはLangChainの開発の手間を簡単にするラッパー関数です。LCELでのチェイン構築を __簡単、直感的に__ します。開発生産性を向上させるため、開発モードやデータ検証などを入れ、一連の構築作業を簡単にします。

## SnappyChainの特徴

SnappyChainの何よりの特徴は、まずは直感的・簡単にLangChainの処理を構築できるということです。

### 特徴1. 簡単・直感的なチェイン構築

__従来のLangChain__

以下は、LangChainを使用したLCELでのシステムプロンプトとユーザープロンプトを設定し、文字列を取り出す例です。LCELによって、| でチェインを繋ぐことができます。


```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser


# プロンプトテンプレートを作成
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("あなたは知識豊富なアシスタントです。簡潔かつ正確に回答してください。"),
    HumanMessagePromptTemplate.from_template("{question}")
])


# LLM（OpenAI GPT-4）を設定
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

# 出力パーサー
output_parser = StrOutputParser()

# LCEL を利用して、プロンプト → LLM → 出力パース の流れを構築
chain = prompt | llm | output_parser

# ユーザーの入力
user_input = {"question": "量子コンピュータとは何ですか？"}

# 実行
response = chain.invoke(user_input)
```

__SnappyChainの場合__

同じコードをSnappyChainで書きます。SnappyChainを使うことで直感的に記載できます。また記憶負担を高めるように務めた関数名を採用します。

```python
from snappychain import system_prompt, human_prompt, openai_chat , output
chain = (
    system_prompt("あなたは知識豊富なアシスタントです。簡潔かつ正確に回答してください")  \
    | human_prompt("{question}")  \
    | openai_chat(model="gpt-4", temperature=0.7)  \
    | output()
)

# ユーザーの入力
user_input = {"question": "量子コンピュータとは何ですか？"}

# 実行
response = chain.invoke(user)
```

### 特徴2.開発モード

開発モード dev()という関数を設けています。dev関数をチェインの冒頭で呼ぶことで関数はデバッグログを色分けをしながら、ログ表示をします。なので、どういった受け答えをしているかを見えやすくします。このログには[onelogger](http://t6^:/)を使っています。

```python
dev()
  | system_prompt("あなたは有能なアシスタントです。")　\
  | user_prompt("{question}") \
  | openai_chat("gpt-4o-mini") \
  | output()
....
```


## インストール 
SnappyChainをインストールするにはPyPIからインストールしてください。

```bash
> pip install snappychain
```

## 使い方 
SnappyChainを使用するには、必要なモジュールをインポートし、データを処理します。
ほとんどLangChainの名前を引き継いでいるので、連想つきやすいと思います。

```python
from snappychain import output

result = output("json")
print(result)
```

### サポート範囲

LangChainの以下の内容までサポートされています。プルリクやサポートのリクエストをお願いします。

__LLM関係__

Streamはまだ対応していません。

|カテゴリ|役割|サポート範囲|補足|
|:--|:--|:--|:--|
|PromptTemplate| |System/Human/AI| |
|Chatモデル|各プロバイダーのチャット用モデル|OpenAI/Ollama/Anthropic/Gemini|※追加でsnappychain-openai, snappychain-ollama, snapchain-anthropic, snapchain-geminiをインストール|
|ResponseSchema|LLMに対し出力時の形式を指定する、構造化出力を使用する||
|OutputParser|LLMのレスポンスをパースする|

__RAG関係__

|カテゴリ|役割|サポート範囲|補足|
|:--|:--|:--|:--|
|DocumentLoader|AzureAIdocument| |基本的に有償・無償のローダを一通りという観点|
|DocumentSplitter|文書の分割| | |
|Embeddings|文書をEmbeddingsに変換する| | |
|VectorStore|VectoreS| | |

* LangChainローダー
  - DrirectoryLoader: 拡張子に応じてロードする。Azureが有効の場合は、有償のAzureを使用する。
  - TextLoader: テキストファイル用のローダ // ローダ for text files
  - CSVLoader: CSVファイル用のローダ // CSVファイル用のローダ
  - PyPDFLoader: PDFファイル用のローダ // PDFファイル用のローダ
  - AzureAIDocumentLoader: Azure AI Document 用のローダ // ローダ for Azure AI Document(有償だが)
  - MarkItDwonItLodear: MarkItDownを使ったOffice文書ローダ

* LangChain Splitter
  - CharacterTextSplitter: 文字列を分割する // 文字列を分割する
  - MarkdownTextSplitter: Markdownを使った分割 // Markdownを使った分割
  - JSONTextSplitter: JSONを使った分割 // JSONを使った分割
  - PythonTextSplitter: Pythonを使った分割 // Pythonを使った分割

* Embeddings
  - OpenAIEmbeddings: OpenAIのEmbeddingsを使う // OpenAIのEmbeddingsを使う
  - OllamaEmbeddings: OllamaのEmbeddingsを使う // OllamaのEmbeddingsを使う

* VectorStore
  - FAISSVectorStore: FAISSを使う // FAISSを使う
  - ChromaVectorStore: Chromaを使う // Chromaを使う

### エージェント/Rag


* Rag用ユーティリティ
  - rag_start_up(): RAGスタートアップ // RAGスタートアップ
  - 
  - rag_query(): RAGクエリ // RAGクエリ


## 明示的モジュラリティ

将来定期にChat、DocumentLoaderなど外部APIが必要になる技術は初期時にインターフェースのみ有効化されています。importして有効になるようにします。拡張のパッケージをimportすることで利用が有効になります。

```python
from snappychat import openai_chat
openai_chat()  " Module Not Found Error

```

```python
from snappychat import openai_chat
import snappychat-openai  # モジュールが有効化

openai_chat()  # 利用可能
```



## 貢献 
貢献を歓迎します！詳細については、[貢献ガイドライン](CONTRIBUTING.md)をお読みください。

## ライセンス 
このプロジェクトはMITライセンスの下でライセンスされています - 詳細については[LICENSE](LICENSE)ファイルを参照してください。

## サポート 
質問がある場合やサポートが必要な場合は、GitHubで問題をオープンしてください。

---

## 現在サポートしている環境
- Python 3.8以上
- Windows, macOS, Linux

## メリット
SnappyChainは、データ処理を簡単にし、作業効率を向上させるために設計されています。使いやすいインターフェースとカスタマイズ可能な出力により、ユーザーは自分のニーズに合わせてツールを調整できます。

## LangChain Splitter


