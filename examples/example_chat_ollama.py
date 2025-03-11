from snappychain import system_prompt, human_prompt
from snappychain import ollama_chat, output
from snappychain import schema
from oneenv import load_dotenv

from langchain.globals import set_verbose

# verboseモードを有効にする
# Enable verbose mode
set_verbose(True)

load_dotenv()

# メイン関数
# Main function
def main():
    # チェインを作成
    # Create a chain
    chain = system_prompt("""あなたは山に関する情報を提供するアシスタントです。
山に関する日本語での質問を受け取り、日本語で情報を提供してください。

必ず以下の形式で日本語で回答してください：
- 場所: [場所の説明]
- 標高: [標高（メートル）、数字のみ]

それ以外のテキストや説明は追加しないでください。""") \
        | human_prompt("{question}") \
        | schema([
            {
                "name":"場所",
                "description": "山の場所"
            },
            {
                "name":"標高",
                "description": "山の標高(m)"
            }
        ])\
        | ollama_chat(model="phi4-mini:latest", temperature=0.2) \
        | output()
    
    # 基本的な呼び出し
    # Basic call
    result = chain.invoke({"question":"富士山について教えて"}, verbose=True)
    
    print("応答結果 / Response result:")
    print(result)

if __name__ == "__main__":
    main() 