# SnappyChain コンセプト

## 概要

- **SnappyChainは、PythonのLangChainのラッパーです。**
- **チェーンを実現する関数を提供し、直感的にLCELを利用できるようにします。**
- **コンポーネントレジストリによるキャッシュ機能で、効率的な実行を実現します。**

## チェーン構文

- **例:**

  ```python
  result = system("環境を初期化") | user() | openai_chat() | schema() | invoke()
  print(result)
  ```

- **この構文は、複数の関数をパイプライン形式でチェーンでき、可読性と使いやすさを向上させます。**

## コンポーネントレジストリ

SnappyChainでは、モデルとプロンプトテンプレートを効率的に管理するためのコンポーネントレジストリを実装しています。

### 主な特徴

- **シングルトンパターン**: アプリケーション全体で単一のインスタンスを保証します。
- **自動識別子生成**: モデルパラメータやプロンプトから短い識別子（8文字）を自動生成します。
- **キャッシュ機能**: 生成された識別子をキーとして、モデルとプロンプトテンプレートをキャッシュします。
- **詳細出力モード**: デバッグやトレース用に詳細な出力を設定できます。

### メリット

- **パフォーマンス向上**: モデルやプロンプトテンプレートの再作成を避けることで、実行速度が向上します。
- **メモリ効率**: 同じモデルやプロンプトを複数回作成せず、メモリ使用量を削減します。
- **一貫性**: 同じパラメータで作成されたモデルは常に同じインスタンスを参照します。

## 提供する関数候補一覧

### プロンプト用関数

| 関数 | 説明 |
|----------|----------------|
| system_prompt() | システムプロンプトを実行する。 |
| human_prompt() | ユーザープロンプトを取得または促す。 |
| ai_prompt() | ユーザープロンプトを取得または促す。 |

### モデル用関数

| 関数 | 説明 |
|----------|----------------|
| openai_chat() | OpenAI APIと連携し、言語タスクを実行する。 |
| gemini_chat() | Google Gemini APIと連携し、言語タスクを実行する。 |
| ollama_chat() | Ollama APIと連携し、言語タスクを実行する。 |
| anthropic_chat() | Anthropic APIと連携し、言語タスクを実行する。 |
| huggingface_chat() | Hugging FaceのTransformersモデルを利用する。 |
| cohere_chat() | Cohereの言語モデルを利用する。 |
| azure_openai_chat() | Azure上でホストされたOpenAIモデルを利用する。 |
| replicate_chat() | Replicateプラットフォーム上のモデルを利用する。 |
| custom_model_chat() | カスタムモデルやローカルでホストされたモデルを利用する。 |
| schema() | 出力を指定されたスキーマに設定する。 |

## サンプルコード

```python
# チェーン利用例
result = system("環境を初期化") | user() | openai_chat() | schema() | invoke()
print(result)
```

## 設計パターン

RunnableLambdaを使用して、辞書型の引数dataを受け取り、辞書型を返却することで、チェーンに必要なデータの授受を行います。

```
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# システムプロンプトを設定
def system(text: str):
    return RunnableLambda(lambda data: {**data, "system": text})

# ユーザープロンプトを設定
def user(text: str):
    return RunnableLambda(lambda data: {**data, "user": text})

# OpenAIを呼び出す（データ共有）
def openai(model: str = "gpt-4o"):
    llm = ChatOpenAI(model=model)

    def process(data):
        prompt = f"System: {data['system']}\nUser: {data['user']}"
        return llm.invoke(prompt)

    return RunnableLambda(process)

# LCELチェーンを作成（データを共有）
chain = (
    system("あなたは有能なアシスタントです。")
    | user("こんにちは！元気？")
    | openai()
)
```

### dataオブジェクトに保持する

|属性| 内容| 例 |
|---|---|---|
| prompt |プロンプトの配列| [{"system": "あなたは有能なアシスタントです。"}, {"user": "こんにちは！元気？"}] |
| model | 使用するチャットモデル |ChatOpenAI(model="gpt-4o")|
| schema | 出力を指定されたスキーマに設定する。 | {スキーマのJSON} |
| tool | 関数モデル| [{'name': 'move_file',
 'description': 'Move or rename a file from one location to another',
 'parameters': {'type': 'object',
  'properties': {'source_path': {'description': 'Path of the file to move',
    'type': 'string'},
   'destination_path': {'description': 'New path for the moved file',
    'type': 'string'}},
  'required': ['source_path', 'destination_path']}}]|


## その他の観点

- **モジュール設計：各関数が単一責任原則に従うことでシンプルさを保っています。**
- **表現力豊かなインタフェース：複雑なワークフローをシンプルなAPIで実現します。**
- **デバッグ容易性と拡張性：関心事の分離により、テストや拡張がしやすい設計です。**
- **効率的なリソース管理：コンポーネントレジストリによるキャッシュ機能で、リソースを効率的に利用します。**

## 結論

- **SnappyChainはチェーンの構築と実行を簡素化し、ボイラープレートコードを削減するとともに、開発者の生産性を向上させます。**
- **コンポーネントレジストリによるキャッシュ機能で、実行速度の向上とメモリ使用量の削減を実現します。**