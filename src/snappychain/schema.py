# shema.py
# This module contains a function to construct a ResponseSchema from langchain.
# このモジュールはlangchainのResponseSchemaを生成する関数を含みます。

from langchain.output_parsers.structured import ResponseSchema
from langchain_core.runnables import RunnableLambda
from .chain import Chain, get_chain_id
from .print import verbose_print, Color
# Importing ResponseSchema from langchain
# langchainからResponseSchemaをインポートします。

def schema(schema_list: list[dict]) -> Chain:
    """
    Construct a ResponseSchema object with the provided name and description.
    指定された名前と説明をもとにResponseSchemaオブジェクトを作成する関数です。

    Parameters:
        name (str): The name identifier of the schema.
                    スキーマの名前（識別子）。
        description (str): A detailed description for the schema.
                           スキーマの詳細な説明。
        type (str): type of the variable.
                    変数の型情報

    Returns:
        Chain: The constructed Chain object.
                作成されたChainオブジェクト。
    """
    # Create a ResponseSchema object using the provided name and description
    # 提供された名前と説明を使用してResponseSchemaオブジェクトを作成します。
    def inner(data, *args, **kwargs):
        if "_session" not in data:
            data["_session"] = {}
        
        # verboseパラメータを処理
        # Process verbose parameter
        verbose_mode = kwargs.get("verbose", False)
        
        # チェインIDとインデックスを取得
        chain_id = None
        if "chain" in data["_session"]:
            chain = data["_session"].get("chain")
            chain_id = get_chain_id(chain)
        
        if verbose_mode:
            # デバッグ情報を表示
            verbose_print("デバッグ情報 / Debug info", {
                "入力データ / Input data": data,
                "引数 / Args": args,
                "キーワード引数 / Kwargs": kwargs
            }, Color.MAGENTA)
            
            # 1. チェインの開始を表示
            verbose_print("スキーマ設定開始 / Schema setting start", 
                       f"Chain ID: {chain_id}", Color.CYAN)
            
            # 2. リクエストの全体を表示
            verbose_print("スキーマ定義 / Schema definition", schema_list, Color.GREEN)
        
        # スキーマオブジェクトを作成
        # Create schema objects
        schema_objects = []
        for schema in schema_list:
            name = schema["name"]
            description = schema["description"]
            if "type" in schema:
                type = schema["type"]
            else:
                type = "text"
            schema_object = ResponseSchema(name=name, description=description, type=type)
            schema_objects.append(schema_object)
        
        # セッションにスキーマを保存
        # Save schema to session
        session = data["_session"]
        session["schema"] = schema_objects
        
        # システムプロンプトにフォーマット指示を追加
        # Add format instructions to system prompt
        if "prompt" in session:
            from langchain.output_parsers.structured import StructuredOutputParser
            parser = StructuredOutputParser.from_response_schemas(schema_objects)
            format_instructions = parser.get_format_instructions()
            
            # 中括弧をエスケープ（二重中括弧にする）
            # Escape curly braces by doubling them
            format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
            
            # システムプロンプトを探して、フォーマット指示を追加
            # Find system prompt and add format instructions
            for prompt in session["prompt"]:
                if "system" in prompt:
                    prompt["system"] += f"\n\n{format_instructions}"
                    break
            else:
                # システムプロンプトがない場合は追加
                # Add system prompt if not exists
                session["prompt"].insert(0, {"system": format_instructions})
        
        if verbose_mode:
            # 3. 応答アウトプットを表示
            verbose_print("スキーマ設定結果 / Schema setting result", 
                       {"schemas": [{"name": s.name, "description": s.description, "type": s.type} 
                                  for s in schema_objects]}, Color.YELLOW)
            
            # 4. チェインの終了を表示
            verbose_print("スキーマ設定完了 / Schema setting complete", {
                "chain_id": chain_id,
                "format_instructions": format_instructions if "prompt" in session else None
            }, Color.CYAN)
        
        # argsとkwargsをセッションに保存して後続のチェインに伝達
        # Save args and kwargs to session to pass to subsequent chains
        if args:
            session["args"] = args
        if kwargs:
            session["kwargs"] = kwargs
            
        return data
    
    return Chain(inner)
