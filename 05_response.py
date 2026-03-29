import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
model = ChatGroq(
    model="qwen/qwen3-32b",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

print("=" * 70)
print("练习 5：理解返回值")
print("=" * 70)

response1 = model.invoke("用20字以内介绍一下什么是人工智能")
print(f"\n输出主要内容:{response1.content}\n")

print(f"\n消息ID : {response1.id}\n")

print(f"\n响应元数据 : {response1.response_metadata}\n")

print(f"\n模型名称 : {response1.response_metadata.get('model_name')} \n")

print(f"\n结束原因 : {response1.response_metadata.get('finish_reason')}\n")

# token使用情况

print(f"\ntoken使用情况 : {response1.response_metadata.get('token_usage',{})}\n")

usage = response1.response_metadata.get('token_usage', {})

# token消耗的详情
print(f"输入token消耗 : {usage.get('prompt_tokens')}\n")
print(f"输出token消耗 : {usage.get('completion_tokens')}\n")
print(f"总计token消耗 : {usage.get('total_tokens')}\n")


print("\n【5. 其他字段】")
print(f"response.type = {response1.type}")
print(f"response.additional_kwargs = {response1.additional_kwargs}")

