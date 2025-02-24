# shema.py
# This module contains a function to construct a ResponseSchema from langchain.
# このモジュールはlangchainのResponseSchemaを生成する関数を含みます。

from langchain.output_parsers.structured import ResponseSchema
from langchain_core.runnables import RunnableLambda
# Importing ResponseSchema from langchain
# langchainからResponseSchemaをインポートします。

def schema(schema_list: list[dict]) -> RunnableLambda:
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
        ResponseSchema: The constructed ResponseSchema object.
                        作成されたResponseSchemaオブジェクト。
    """
    # Create a ResponseSchema object using the provided name and description
    # 提供された名前と説明を使用してResponseSchemaオブジェクトを作成します。
    def inner(data):
        if "schema" not in data["_session"]:
            data["_session"]["schema"] = []
        for schema in schema_list:
            name = schema["name"]
            description = schema["description"]
            if "type" in schema:
                type = schema["type"]
            else:
                type = "text"
            schema_object = ResponseSchema(name=name, description=description, type=type)
            data["_session"]["schema"].append(schema_object)
        return data
    return RunnableLambda(inner)
