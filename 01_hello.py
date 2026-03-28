from dotenv import load_dotenv
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


