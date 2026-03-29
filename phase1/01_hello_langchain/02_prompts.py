from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
import os

load_dotenv()

model = ChatGroq(
    model="qwen/qwen3-32b",  # Groq 上的千问模型
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 方式1
response = model.invoke("你好，请用中文介绍自己")
print(response.content)

print("___________________________\n")


# 方式2 ：字典列表
print("字典列表进行输入")
message2 = [
    {"role" : "system","content" : "你是一个简洁的助手,回答限制在30字以内"},
    {"role" : "user","content" : "Python是什么？"}
]
response2 = model.invoke(message2)
print(response2.content)

print("----------------------------\n")

# 方式3 ：消息对象
print("消息对象方式输入")
message3 = [
    SystemMessage(content="你是个幽默的助手,喜欢用比喻"),
    HumanMessage(content="什么是Python?")
]
response3 = model.invoke(message3)
print(response3.content)



