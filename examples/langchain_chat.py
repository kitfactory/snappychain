from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LLM（OpenAI GPT-4）を設定
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# プロンプトテンプレートを作成
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("あなたは知識豊富なアシスタントです。簡潔かつ正確に回答してください。"),
    HumanMessagePromptTemplate.from_template("{question}")
])

# 出力パーサー
output_parser = StrOutputParser()

# LCEL を利用して、プロンプト → LLM → 出力パース の流れを構築
chain = prompt  | llm | output_parser

# ユーザーの入力
user_input = {"question": "量子コンピュータとは何ですか？"}

# 実行
response = chain.invoke(user_input)

print(response)
