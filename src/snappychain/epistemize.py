# epistemize.py
# 認識論的な手法を用いて、知識を発見する
# 
# LLMのチェインと連結された場合に、以下を実施する
# ユーザーとのやりとりから、言葉：Neologismとやり方：Praxisを発見する
# 

from typing import Dict, List, Any, Optional
from langchain_core.runnables import RunnableLambda
from snappychain.print import verbose_print, debug_print, Color
import os
import json
import threading
import concurrent.futures
from oneenv import oneenv, load_dotenv

# snappychainのモジュールをインポート
from .prompt import system_prompt, human_prompt
from .chat import openai_chat
from .schema import schema
from .output import output
from .devmode import validate as dev

# スレッドプールエグゼキューター
# Thread pool executor
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

@oneenv
def episteme_env_template():
    """
    Define environment variables template for Episteme.
    Epistemeの環境変数テンプレートを定義します。
    
    Returns:
        Dict: Environment variables template for Episteme.
        Dict: Epistemeの環境変数テンプレート。
    """
    return {
        "EPISTEME_DIR": {
            "description": "Directory path for storing Episteme data files. Default is ~/.snappychain",
            "description_ja": "Epistemeデータファイルを保存するディレクトリパス。デフォルトは ~/.snappychain",
            "default": "",
            "required": False
        },
        "EPISTEME_FILENAME": {
            "description": "Filename for Episteme data. Default is episteme.json",
            "description_ja": "Epistemeデータのファイル名。デフォルトは episteme.json",
            "default": "episteme.json",
            "required": False
        }
    }

class Episteme:
    """
    A singleton class to manage neologisms and praxis in a file.
    ファイルで新語と実践を管理するシングルトンクラス。
    
    This class stores and retrieves neologisms and praxis from a JSON file.
    このクラスはJSONファイルから新語と実践を保存および取得します。
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """
        Create a singleton instance of Episteme.
        Epistemeのシングルトンインスタンスを作成します。
        
        Returns:
            Episteme: The singleton instance.
            Episteme: シングルトンインスタンス。
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Episteme, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize the Episteme with a file path.
        ファイルパスでEpistemeを初期化します。
        
        Args:
            file_path (Optional[str], optional): Path to the episteme file. Defaults to None.
            file_path (Optional[str], optional): 認識論ファイルへのパス。デフォルトはNone。
        """
        if self._initialized:
            return
            
        # 環境変数を読み込む
        # Load environment variables
        load_dotenv()
        
        # デフォルトのファイルパスを設定
        # Set default file path
        if file_path is None:
            # 環境変数からディレクトリパスを取得、設定されていない場合はデフォルト値を使用
            # Get directory path from environment variable, use default if not set
            episteme_dir = os.environ.get("EPISTEME_DIR", "")
            if not episteme_dir:
                # ユーザーのホームディレクトリに保存
                # Save to user's home directory
                home_dir = os.path.expanduser("~")
                episteme_dir = os.path.join(home_dir, ".snappychain")
            
            # 環境変数からファイル名を取得、設定されていない場合はデフォルト値を使用
            # Get filename from environment variable, use default if not set
            episteme_filename = os.environ.get("EPISTEME_FILENAME", "episteme.json")
            
            self.file_path = os.path.join(episteme_dir, episteme_filename)
        else:
            self.file_path = file_path
            
        # ディレクトリが存在しない場合は作成
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        
        # 初期データを設定
        # Set initial data
        self.data = {
            "neologisms": [],
            "praxis": []
        }
        
        # ファイルから読み込み
        # Load from file
        self.load()
        
        self._initialized = True
    
    def load(self) -> None:
        """
        Load knowledge data from the file.
        ファイルから知識データを読み込みます。
        """
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    # データ構造を検証
                    # Validate data structure
                    if isinstance(loaded_data, dict) and "neologisms" in loaded_data and "praxis" in loaded_data:
                        self.data = loaded_data
                        verbose_print("Episteme", f"Loaded {len(loaded_data['neologisms'])} neologisms and {len(loaded_data['praxis'])} praxis", Color.GREEN)
                    else:
                        verbose_print("Episteme", "Invalid data structure in episteme file. Using default.", Color.YELLOW)
        except Exception as e:
            verbose_print("Episteme", f"Error loading episteme: {str(e)}", Color.RED)
    
    def save(self) -> None:
        """
        Save knowledge data to the file.
        知識データをファイルに保存します。
        """
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            verbose_print("Episteme", f"Saved {len(self.data['neologisms'])} neologisms and {len(self.data['praxis'])} praxis", Color.GREEN)
        except Exception as e:
            verbose_print("Episteme", f"Error saving episteme: {str(e)}", Color.RED)
    
    def add_neologisms(self, new_neologisms: List[Dict[str, str]]) -> None:
        """
        Add new neologisms to the store.
        新しい新語をストアに追加します。
        
        Args:
            new_neologisms (List[Dict[str, str]]): List of new neologisms to add.
            new_neologisms (List[Dict[str, str]]): 追加する新語のリスト。
        """
        # 既存の用語を確認
        # Check existing terms
        existing_terms = {item["term"]: i for i, item in enumerate(self.data["neologisms"])}
        
        for neologism in new_neologisms:
            term = neologism.get("term")
            if not term:
                continue
                
            # 既存の用語がある場合は更新、なければ追加
            # Update if term exists, otherwise add
            if term in existing_terms:
                self.data["neologisms"][existing_terms[term]] = neologism
                debug_print("Episteme", f"Updated neologism: {term}", Color.CYAN)
            else:
                self.data["neologisms"].append(neologism)
                debug_print("Episteme", f"Added new neologism: {term}", Color.CYAN)
        
        # 変更をファイルに保存
        # Save changes to file
        self.save()
    
    def add_praxis(self, new_praxis: List[Dict[str, str]]) -> None:
        """
        Add new praxis to the store.
        新しい実践をストアに追加します。
        
        Args:
            new_praxis (List[Dict[str, str]]): List of new praxis to add.
            new_praxis (List[Dict[str, str]]): 追加する実践のリスト。
        """
        # 既存の指示を確認
        # Check existing instructions
        existing_instructions = {item["instruction"]: i for i, item in enumerate(self.data["praxis"])}
        
        for praxis in new_praxis:
            instruction = praxis.get("instruction")
            if not instruction:
                continue
                
            # 既存の指示がある場合は更新、なければ追加
            # Update if instruction exists, otherwise add
            if instruction in existing_instructions:
                self.data["praxis"][existing_instructions[instruction]] = praxis
                debug_print("Episteme", f"Updated praxis: {instruction}", Color.MAGENTA)
            else:
                self.data["praxis"].append(praxis)
                debug_print("Episteme", f"Added new praxis: {instruction}", Color.MAGENTA)
        
        # 変更をファイルに保存
        # Save changes to file
        self.save()
    
    def get_neologisms(self) -> List[Dict[str, str]]:
        """
        Get all neologisms from the store.
        ストアからすべての新語を取得します。
        
        Returns:
            List[Dict[str, str]]: List of all neologisms.
            List[Dict[str, str]]: すべての新語のリスト。
        """
        debug_print("Episteme", f"Retrieved {len(self.data['neologisms'])} neologisms", Color.CYAN)
        return self.data["neologisms"]
    
    def get_praxis(self) -> List[Dict[str, str]]:
        """
        Get all praxis from the store.
        ストアからすべての実践を取得します。
        
        Returns:
            List[Dict[str, str]]: List of all praxis.
            List[Dict[str, str]]: すべての実践のリスト。
        """
        debug_print("Episteme", f"Retrieved {len(self.data['praxis'])} praxis", Color.MAGENTA)
        return self.data["praxis"]

def epistemize(model: str = "gpt-4o-mini", temperature: float = 0.2, file_path: Optional[str] = None) -> RunnableLambda:
    """
    Extract knowledge terms (Neologism) and action instructions (Praxis) from chat responses using LLM.
    LLMを使用してチャットレスポンスから知識用語（Neologism）と実働指示（Praxis）を抽出します。

    This function can be connected with chat.py and LCEL (LangChain Expression Language).
    この関数はchat.pyおよびLCEL（LangChain Expression Language）と連結できます。

    Args:
        model (str, optional): The LLM model to use. Defaults to "gpt-4o-mini".
        temperature (float, optional): The temperature parameter for the LLM. Defaults to 0.2.
        file_path (Optional[str], optional): Path to the episteme file. Defaults to None.

    Returns:
        RunnableLambda: A runnable lambda that can be used in a LangChain pipeline.
        RunnableLambda: LangChainパイプラインで使用できる実行可能なラムダ関数。
    """
    # 認識論のインスタンスを作成
    # Create episteme instance
    episteme = Episteme(file_path)
    
    def process_epistemize(response_content: str, dev_mode: bool = False) -> Dict[str, Any]:
        """
        Process the epistemize function in a separate thread.
        別スレッドでepistemize関数を処理します。
        
        Args:
            response_content (str): The response content to process.
            response_content (str): 処理するレスポンスの内容。
            dev_mode (bool, optional): Whether to enable development mode. Defaults to False.
            dev_mode (bool, optional): 開発モードを有効にするかどうか。デフォルトはFalse。
            
        Returns:
            Dict[str, Any]: The epistemize result containing neologisms and praxis.
            Dict[str, Any]: 新語と実践を含むepistemizeの結果。
        """
        try:
            # snappychainのチェインを作成して実行
            # Create and execute snappychain chain
            epistemize_chain = dev() \
                | system_prompt("""
                You are an epistemological analyzer that identifies and extracts two types of knowledge from text:
                
                1. Neologisms: New terms, concepts, or vocabulary that represent knowledge or ideas.
                2. Praxis: Action instructions, practical applications, or methodologies.
                
                Analyze the following chat response and extract all neologisms and praxis found within it.
                Be thorough but concise in your extraction.
                """) \
                | human_prompt("Chat Response to analyze:\n\n{response_content}") \
                | schema([
                    {
                        "name": "neologisms",
                        "description": "List of new terms or concepts (Neologism) identified in the chat response",
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "term": {"type": "string", "description": "The term or concept"},
                                "definition": {"type": "string", "description": "Brief definition or explanation of the term"}
                            },
                            "required": ["term", "definition"]
                        }
                    },
                    {
                        "name": "praxis",
                        "description": "List of action instructions or practical applications (Praxis) identified in the chat response",
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "instruction": {"type": "string", "description": "The action instruction or practical application"},
                                "context": {"type": "string", "description": "Context or situation where this instruction applies"}
                            },
                            "required": ["instruction", "context"]
                        }
                    }
                ]) \
                | openai_chat(model=model, temperature=temperature)
            
            # チェインを実行
            # Execute the chain
            chain_result = epistemize_chain.invoke({"response_content": response_content, "_dev": dev_mode})
            
            # 構造化レスポンスの形式を確認
            # Check the format of the structured response
            if "structured_response" in chain_result:
                structured_response = chain_result["structured_response"]
                
                # 新語と実践を取得
                # Get neologisms and praxis
                neologisms = structured_response.get("neologisms", [])
                praxis = structured_response.get("praxis", [])
                
                # 認識論に追加
                # Add to episteme
                episteme.add_neologisms(neologisms)
                episteme.add_praxis(praxis)
                
                # 開発モードの時はログ表示
                # Display log in development mode
                if dev_mode:
                    verbose_print("Epistemize", f"Structured Response: {len(neologisms)} neologisms, {len(praxis)} praxis", Color.GREEN)
                
                return {
                    "neologisms": neologisms,
                    "praxis": praxis
                }
            else:
                verbose_print("Epistemize", "Invalid structured response format", Color.RED)
                # エラー時はテスト用のデータを設定
                # Set test data on error
                return {
                    "neologisms": [
                        {
                            "term": "エラー発生",
                            "definition": "エラーが発生しました"
                        }
                    ],
                    "praxis": [
                        {
                            "instruction": "エラーログを確認",
                            "context": "エラーが発生した場合"
                        }
                    ]
                }
                
        except Exception as e:
            verbose_print("Epistemize", f"Error in epistemize thread: {str(e)}", Color.RED)
            # エラー時はテスト用のデータを設定
            # Set test data on error
            return {
                "neologisms": [
                    {
                        "term": "エラー発生",
                        "definition": f"エラー: {str(e)}"
                    }
                ],
                "praxis": [
                    {
                        "instruction": "エラーログを確認",
                        "context": "エラーが発生した場合"
                    }
                ]
            }
    
    def inner(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inner function that processes the data and extracts neologisms and praxis.
        データを処理し、新語（Neologism）と実践（Praxis）を抽出する内部関数。

        Args:
            data (Dict[str, Any]): The input data dictionary containing chat response.
            data (Dict[str, Any]): チャットレスポンスを含む入力データ辞書。

        Returns:
            Dict[str, Any]: The updated data dictionary with extracted neologisms and praxis.
            Dict[str, Any]: 抽出された新語と実践が追加された更新データ辞書。
        """
        # セッションデータを取得
        # Get session data
        session = data.get("_session", {})
        
        # レスポンスが存在するか確認
        # Check if response exists
        response = session.get("response")
        if not response:
            verbose_print("Epistemize", "No response found in session data", Color.RED)
            return data
        
        # レスポンスの内容を取得
        # Get response content
        response_content = response.content if hasattr(response, "content") else str(response)
        
        # 開発モードの場合はログを出力
        # Output log in development mode
        dev_mode = data.get("_dev", False)
        if dev_mode:
            verbose_print("Epistemize", f"Epistemize processing response: {response_content[:100]}...", Color.GREEN)
        
        # 初期値を設定（非同期処理の結果が返ってくるまでの間に表示するデータ）
        # Set initial values (data to display while waiting for async processing)
        session["epistemize"] = {
            "neologisms": [],
            "praxis": [],
            "processing": True  # 処理中フラグ
        }
        
        # 別スレッドで処理を実行
        # Execute processing in a separate thread
        future = _executor.submit(process_epistemize, response_content, dev_mode)
        
        # 非同期処理の完了時のコールバック関数
        # Callback function for when async processing completes
        def on_complete(future):
            try:
                result = future.result()
                # セッションデータを更新（スレッドセーフな方法で）
                # Update session data (in a thread-safe way)
                with threading.Lock():
                    if "_session" in data and "epistemize" in data["_session"]:
                        data["_session"]["epistemize"] = {
                            "neologisms": result["neologisms"],
                            "praxis": result["praxis"],
                            "processing": False  # 処理完了
                        }
                        if dev_mode:
                            verbose_print("Epistemize", "Epistemize processing completed and session updated", Color.GREEN)
            except Exception as e:
                verbose_print("Epistemize", f"Error in epistemize callback: {str(e)}", Color.RED)
        
        # コールバック関数を設定
        # Set callback function
        future.add_done_callback(on_complete)
        
        # 処理を待たずに即座にデータを返す
        # Return data immediately without waiting for processing
        return data
    
    return RunnableLambda(inner)

# テスト用コード
if __name__ == "__main__":
    print("epistemize.pyのテスト実行")
    try:
        # 認識論のインスタンスを直接使用してテスト
        episteme = Episteme()
        print(f"認識論の初期化完了: {episteme.file_path}")
        
        # テスト用の新語と実践を追加
        test_neologisms = [
            {
                "term": "量子ビット",
                "definition": "量子コンピュータの基本単位。0と1の状態を同時に保持できる量子力学的な性質を持つ。"
            },
            {
                "term": "量子重ね合わせ",
                "definition": "量子ビットが複数の状態を同時に取ることができる量子力学的な現象。"
            }
        ]
        
        test_praxis = [
            {
                "instruction": "量子アルゴリズムの選択",
                "context": "問題の性質に応じて、ShorのアルゴリズムやGroverのアルゴリズムなど適切な量子アルゴリズムを選択する。"
            },
            {
                "instruction": "量子回路の設計",
                "context": "解決したい問題を量子ゲートの組み合わせで表現し、効率的な量子回路を設計する。"
            }
        ]
        
        # 認識論に追加
        print("テスト用の新語と実践を追加中...")
        episteme.add_neologisms(test_neologisms)
        episteme.add_praxis(test_praxis)
        
        # 認識論から取得して表示
        print("\n=== 認識論から取得した新語 ===")
        for item in episteme.get_neologisms():
            print(f"- {item['term']}: {item['definition']}")
        
        print("\n=== 認識論から取得した実践 ===")
        for item in episteme.get_praxis():
            print(f"- {item['instruction']} (適用文脈: {item['context']})")
        
        # 非同期処理のテスト
        print("\n=== 非同期処理のテスト ===")
        # テスト用のデータを作成
        test_data = {
            "_session": {
                "response": {
                    "content": "量子コンピュータは、量子力学の原理を利用して計算を行うコンピュータです。従来のコンピュータが0と1のビットを使用するのに対し、量子コンピュータは量子ビット（キュービット）を使用します。量子コンピュータは特定の問題に対して指数関数的な高速化を実現できる可能性があります。"
                }
            },
            "_dev": True
        }
        
        # epistemize関数を実行
        print("epistemize関数を実行中...")
        epistemize_fn = epistemize(file_path=episteme.file_path)
        result = epistemize_fn.invoke(test_data)
        
        # 初期状態を確認
        print("\n初期状態（非同期処理開始直後）:")
        if "_session" in result and "epistemize" in result["_session"]:
            epistemize_data = result["_session"]["epistemize"]
            print(f"処理中フラグ: {epistemize_data.get('processing', False)}")
            print(f"新語数: {len(epistemize_data.get('neologisms', []))}")
            print(f"実践数: {len(epistemize_data.get('praxis', []))}")
        else:
            print("epistemizeデータが見つかりません")
        
        # 非同期処理の完了を待つ
        print("\n非同期処理の完了を待っています...")
        import time
        # 最大5秒待機
        for i in range(10):
            time.sleep(0.5)
            if "_session" in result and "epistemize" in result["_session"]:
                if not result["_session"]["epistemize"].get("processing", False):
                    print(f"処理が完了しました（{(i+1)*0.5}秒後）")
                    break
            print(".", end="", flush=True)
        print()
        
        # 処理完了後の状態を確認
        print("\n処理完了後の状態:")
        if "_session" in result and "epistemize" in result["_session"]:
            epistemize_data = result["_session"]["epistemize"]
            print(f"処理中フラグ: {epistemize_data.get('processing', False)}")
            print(f"新語数: {len(epistemize_data.get('neologisms', []))}")
            print(f"実践数: {len(epistemize_data.get('praxis', []))}")
            
            if len(epistemize_data.get('neologisms', [])) > 0:
                print("\n抽出された新語:")
                for item in epistemize_data["neologisms"]:
                    print(f"- {item['term']}: {item['definition']}")
            
            if len(epistemize_data.get('praxis', [])) > 0:
                print("\n抽出された実践:")
                for item in epistemize_data["praxis"]:
                    print(f"- {item['instruction']} (適用文脈: {item['context']})")
        else:
            print("epistemizeデータが見つかりません")
        
        # 認識論の最終状態を表示
        print("\n=== 認識論の最終状態 ===")
        print(f"保存先: {episteme.file_path}")
        print(f"新語数: {len(episteme.get_neologisms())}")
        print(f"実践数: {len(episteme.get_praxis())}")
        
        # スレッドプールをシャットダウン
        print("\nスレッドプールをシャットダウンしています...")
        _executor.shutdown(wait=True)
        print("テスト完了")
        
    except Exception as e:
        import traceback
        print(f"テストエラー: {e}")
        traceback.print_exc()
