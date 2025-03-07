"""
Custom OpenAI Chat implementation with verbose output using RunnableLambda
RunnableLambdaを使用した詳細出力付きのカスタムOpenAIチャット実装
"""

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
import os
import json
import sys
import io

# ログファイルの設定
# Log file configuration
log_file_path = "study/verbose_output.log"
log_file = open(log_file_path, "w", encoding="utf-8")

# 環境変数からAPIキーを取得（設定されていない場合は警告を表示）
# Get API key from environment variable (show warning if not set)
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("警告: OPENAI_API_KEYが設定されていません。")
    print("Warning: OPENAI_API_KEY is not set.")
    log_file.write("警告: OPENAI_API_KEYが設定されていません。\n")
    log_file.write("Warning: OPENAI_API_KEY is not set.\n")

# カスタムのデバッグ関数
# Custom debug function
def debug_print(prefix, data, verbose=False):
    """
    デバッグ情報を出力する関数
    Function to output debug information
    
    Args:
        prefix (str): 出力の接頭辞 / Output prefix
        data (any): 出力するデータ / Data to output
        verbose (bool): 詳細出力モードかどうか / Whether in verbose mode
    """
    if verbose:
        separator = f"\n{'='*20} {prefix} {'='*20}"
        print(separator)
        log_file.write(f"{separator}\n")
        
        if isinstance(data, dict) or isinstance(data, list):
            json_data = json.dumps(data, ensure_ascii=False, indent=2)
            print(json_data)
            log_file.write(f"{json_data}\n")
        else:
            print(data)
            log_file.write(f"{data}\n")
        
        end_separator = f"{'='*50}\n"
        print(end_separator)
        log_file.write(f"{end_separator}\n")
        
        # ファイルをフラッシュして確実に書き込む
        # Flush the file to ensure writing
        log_file.flush()

# カスタムのOpenAIチャット関数
# Custom OpenAI chat function
def custom_openai_chat(data, model_name="gpt-4o-mini", temperature=0.7, verbose=False):
    """
    OpenAIのチャットモデルを使用してレスポンスを生成するカスタム関数
    Custom function to generate responses using OpenAI's chat model
    
    Args:
        data (dict): 入力データ / Input data
        model_name (str): 使用するモデル名 / Model name to use
        temperature (float): 温度パラメータ / Temperature parameter
        verbose (bool): 詳細出力モードかどうか / Whether in verbose mode
        
    Returns:
        str: 生成されたレスポンス / Generated response
    """
    # verboseパラメータを取得（configに格納されている場合）
    # Get verbose parameter (if stored in config)
    if hasattr(data, "get") and callable(data.get):
        config = data.get("config", {})
        if isinstance(config, dict):
            verbose = config.get("verbose", verbose)
    
    debug_print("入力データ / Input Data", data, verbose)
    
    # 処理開始時間を記録
    # Record processing start time
    start_time = time.time()
    
    # OpenAIのチャットモデルを初期化
    # Initialize OpenAI chat model
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    debug_print("使用モデル / Model Used", f"{model_name} (temperature={temperature})", verbose)
    
    # レスポンスを生成
    # Generate response
    response = llm.invoke(data)
    
    # 処理時間を計算
    # Calculate processing time
    processing_time = time.time() - start_time
    
    debug_print("生成されたレスポンス / Generated Response", response.content, verbose)
    debug_print("処理時間 / Processing Time", f"{processing_time:.2f}秒 / seconds", verbose)
    
    return response.content

# RunnableLambdaでラップしたカスタム関数
# Custom function wrapped in RunnableLambda
def create_custom_chain(model_name="gpt-4o-mini", temperature=0.7):
    """
    カスタムチェインを作成する関数
    Function to create a custom chain
    
    Args:
        model_name (str): 使用するモデル名 / Model name to use
        temperature (float): 温度パラメータ / Temperature parameter
        
    Returns:
        Chain: 作成されたチェイン / Created chain
    """
    # システムプロンプトとユーザープロンプトを定義
    # Define system prompt and user prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("あなたは知識豊富なアシスタントです。簡潔かつ正確に回答してください。"),
        HumanMessagePromptTemplate.from_template("{question}")
    ])
    
    # 出力パーサー
    # Output parser
    output_parser = StrOutputParser()
    
    # 入力処理用のRunnableLambda
    # RunnableLambda for input processing
    def input_processor_func(*args, **kwargs):
        # verboseパラメータを取得
        # Get verbose parameter
        verbose = kwargs.get("verbose", False)
        
        # デバッグ出力
        # Debug output
        debug_print("入力プロセッサ引数 / Input Processor Args", args, verbose)
        debug_print("入力プロセッサキーワード引数 / Input Processor Kwargs", kwargs, verbose)
        
        # 入力データを取得
        # Get input data
        input_data = args[0] if args else kwargs.get("input", {})
        
        # configに格納
        # Store in config
        config = {"verbose": verbose}
        
        # configを追加
        # Add config
        if isinstance(input_data, dict):
            input_with_config = {**input_data, "config": config}
        else:
            input_with_config = input_data
        
        return input_with_config

    input_processor = RunnableLambda(input_processor_func)

    # 出力処理用のRunnableLambda
    # RunnableLambda for output processing
    def output_processor_func(output, **kwargs):
        # verboseパラメータを取得
        # Get verbose parameter
        verbose = kwargs.get("verbose", False)
        
        # デバッグ出力
        # Debug output
        debug_print("出力プロセッサ結果 / Output Processor Result", output, verbose)
        
        # 結果を返す
        # Return result
        return output

    output_processor = RunnableLambda(output_processor_func)

    # メインの処理用RunnableLambda
    # RunnableLambda for main processing
    def main_processor_func(data, **kwargs):
        # verboseパラメータを取得
        # Get verbose parameter
        verbose = kwargs.get("verbose", False)
        
        # データからconfigを取得
        # Get config from data
        config = data.get("config", {}) if isinstance(data, dict) else {}
        
        # configからverboseを取得（優先）
        # Get verbose from config (priority)
        verbose = config.get("verbose", verbose)
        
        # カスタム関数を呼び出し
        # Call custom function
        return custom_openai_chat(data, model_name=model_name, temperature=temperature, verbose=verbose)

    main_processor = RunnableLambda(main_processor_func)
    
    # チェインを構築
    # Build the chain
    chain = input_processor | prompt | main_processor | output_parser | output_processor
    
    return chain

# メイン実行部分
# Main execution part
if __name__ == "__main__":
    try:
        # チェインを作成
        # Create chain
        chain = create_custom_chain()
        
        # ユーザーの入力
        # User input
        user_input = {"question": "量子コンピュータとは何ですか？簡潔に説明してください。"}
        
        print("\n🔍 チェインを実行します... / Running the chain...\n")
        log_file.write("\n🔍 チェインを実行します... / Running the chain...\n\n")
        
        # チェインを実行（verboseパラメータを渡す）
        # Run the chain (pass verbose parameter)
        response = chain.invoke(user_input, verbose=True)
        
        print("\n📝 最終結果 / Final Result:")
        print(f"\n{response}\n")
        
        log_file.write("\n📝 最終結果 / Final Result:\n")
        log_file.write(f"\n{response}\n\n")
        
        print(f"\n✅ 詳細なログは {log_file_path} に保存されました。")
        print(f"\n✅ Detailed logs have been saved to {log_file_path}.")
    
    except Exception as e:
        error_message = f"エラーが発生しました: {str(e)}"
        print(f"\n❌ {error_message}")
        log_file.write(f"\n❌ {error_message}\n")
        import traceback
        traceback_str = traceback.format_exc()
        print(traceback_str)
        log_file.write(f"{traceback_str}\n")
    
    finally:
        # ファイルを閉じる
        # Close the file
        log_file.close()
