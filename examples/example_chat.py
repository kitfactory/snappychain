from snappychain import system_prompt, human_prompt
from snappychain import openai_chat, output, dev, schema
from oneenv import load_dotenv

from langchain.globals import set_verbose

load_dotenv()

set_verbose(True)

# メイン関数
# Main function
def main():

    chain = system_prompt("あなたは有能なアシスタントです。") \
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
        | openai_chat(model="gpt-4o-mini", temperature=0.2) \
        | output()
    
    result = chain.invoke({"question":"富士山について教えて"})

    print(result)

if __name__ == "__main__":
    main()
