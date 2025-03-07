"""
Example of using schema with chat to get structured output.
スキーマを使用してチャットから構造化された出力を取得する例。
"""

import sys
import os
import json

# srcディレクトリをパスに追加して、snappychainモジュールをインポートできるようにします
# Add src directory to path to import snappychain module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from snappychain.schema import schema
from snappychain.chat import openai_chat
from snappychain.output import output
from langchain.globals import set_debug

def example_text_classification():
    """
    Text classification example using schema.
    スキーマを使用したテキスト分類の例。
    """
    print("\n=== テキスト分類の例 / Text Classification Example ===")
    
    # デバッグモードを有効化
    # Enable debug mode
    set_debug(True)
    
    # テキスト分類のスキーマを定義
    # Define schema for text classification
    schema_list = [
        {
            "name": "category",
            "description": "The category of the text (e.g. news, sports, technology, entertainment, etc.)",
            "type": "string"
        },
        {
            "name": "confidence",
            "description": "Confidence score from 0 to 100",
            "type": "integer"
        },
        {
            "name": "reasoning",
            "description": "Brief explanation of why this category was chosen",
            "type": "string"
        }
    ]
    
    # パイプラインを作成
    # Create pipeline
    pipeline = (
        schema(schema_list)
        | openai_chat(model="gpt-4o-mini")
        | output("json")
    )
    
    # テキストサンプル
    # Sample text
    text = "Apple released a new iPhone with advanced AI capabilities, setting a new standard in the smartphone industry."
    
    # パイプラインを実行
    # Run pipeline
    result = pipeline.invoke({
        "_session": {
            "prompt": [
                {
                    "system": "You are a text classification assistant. Classify the given text into an appropriate category."
                },
                {
                    "human": f"Please classify the following text: {text}"
                }
            ]
        }
    }, verbose=True)
    
    # 結果を表示
    # Display result
    print("\n結果 / Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

def example_sentiment_analysis():
    """
    Sentiment analysis example using schema.
    スキーマを使用した感情分析の例。
    """
    print("\n=== 感情分析の例 / Sentiment Analysis Example ===")
    
    # デバッグモードを有効化
    # Enable debug mode
    set_debug(True)
    
    # 感情分析のスキーマを定義
    # Define schema for sentiment analysis
    schema_list = [
        {
            "name": "sentiment",
            "description": "The sentiment of the text (positive, negative, or neutral)",
            "type": "string"
        },
        {
            "name": "score",
            "description": "Sentiment score from -1 (most negative) to 1 (most positive)",
            "type": "number"
        },
        {
            "name": "key_phrases",
            "description": "List of phrases that influenced the sentiment score",
            "type": "list[string]"
        }
    ]
    
    # パイプラインを作成
    # Create pipeline
    pipeline = (
        schema(schema_list)
        | openai_chat(model="gpt-4o-mini")
        | output("json")
    )
    
    # テキストサンプル
    # Sample text
    text = "The service was terrible and the food was cold. However, the staff was very apologetic and offered us a free dessert."
    
    # パイプラインを実行
    # Run pipeline
    result = pipeline.invoke({
        "_session": {
            "prompt": [
                {
                    "system": "You are a sentiment analysis assistant. Analyze the sentiment of the given text."
                },
                {
                    "human": f"Please analyze the sentiment of the following text: {text}"
                }
            ]
        }
    }, verbose=True)
    
    # 結果を表示
    # Display result
    print("\n結果 / Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

def example_entity_extraction():
    """
    Entity extraction example using schema.
    スキーマを使用したエンティティ抽出の例。
    """
    print("\n=== エンティティ抽出の例 / Entity Extraction Example ===")
    
    # デバッグモードを有効化
    # Enable debug mode
    set_debug(True)
    
    # エンティティ抽出のスキーマを定義
    # Define schema for entity extraction
    schema_list = [
        {
            "name": "people",
            "description": "List of people mentioned in the text",
            "type": "list[string]"
        },
        {
            "name": "organizations",
            "description": "List of organizations mentioned in the text",
            "type": "list[string]"
        },
        {
            "name": "locations",
            "description": "List of locations mentioned in the text",
            "type": "list[string]"
        },
        {
            "name": "dates",
            "description": "List of dates mentioned in the text",
            "type": "list[string]"
        }
    ]
    
    # パイプラインを作成
    # Create pipeline
    pipeline = (
        schema(schema_list)
        | openai_chat(model="gpt-4o-mini")
        | output("json")
    )
    
    # テキストサンプル
    # Sample text
    text = "Mark Zuckerberg announced that Meta will open a new office in Tokyo, Japan by December 2024. The announcement was made during a press conference at their headquarters in Menlo Park."
    
    # パイプラインを実行
    # Run pipeline
    result = pipeline.invoke({
        "_session": {
            "prompt": [
                {
                    "system": "You are an entity extraction assistant. Extract entities from the given text."
                },
                {
                    "human": f"Please extract entities from the following text: {text}"
                }
            ]
        }
    }, verbose=True)
    
    # 結果を表示
    # Display result
    print("\n結果 / Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

if __name__ == "__main__":
    print("=== スキーマを使用した構造化出力の例 / Structured Output Examples Using Schema ===")
    
    # 各例を実行
    # Run each example
    example_text_classification()
    example_sentiment_analysis()
    example_entity_extraction()
    
    print("\n=== 全ての例が完了しました / All examples completed ===")