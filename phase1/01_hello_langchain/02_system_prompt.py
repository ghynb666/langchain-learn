import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()
model = ChatGroq(model="qwen/qwen3-32b", groq_api_key=os.getenv("GROQ_API_KEY"))

print("=" * 70)
print("练习 2：系统提示的威力")
print("=" * 70)

question = "什么是递归"
print(f"\n问题:{question}\n")

messages = [
    {"role" : "system","content" : "你是一个严肃的计算机科学教授，回答要学术化、专业化"},
    {"role" : "user","content" : question}
]

response1 = model.invoke(messages)
print(f"\n系统提升词1输出:{response1.content}\n")
