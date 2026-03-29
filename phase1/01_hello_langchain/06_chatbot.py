import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
model = ChatGroq(model="qwen/qwen3-32b", groq_api_key=os.getenv("GROQ_API_KEY"))

print("=" * 70)
print("练习 6：简单聊天机器人")
print("=" * 70)


# 初始化对话
messages = [
    {"role" : "system", "content" : "你是一个友好、幽默的助手"}
]
total_token = 0

# 首轮对话

questions = ["你好" ,"你能做什么","告诉我如何从化工厂转型到AI应用开发","我打算学python"]

for i, question in enumerate(questions,1):
    print(f"\n第{i}轮对话\n")
    print(f"用户的问题: {question}")

    # 整合上下文
    messages.append(
        {"role" : "user","content" : question}
    )

    # 本轮对话输出结果
    response = model.invoke(messages)
    print(f"\n本轮AI输出 : {response.content}")

    # 后将AI输出的结果整合至上下文
    messages.append(
        {"role" : "assistant","content": response.content}
    )

    # 获取本轮消耗的token
    usage = response.response_metadata.get('token_usage', {})
    tokens = usage.get('total_tokens', 0)
    total_token += tokens
    print(f"\n本轮消耗token量 ： {tokens},总的token消耗: {total_token}")

print(f"对话结束，总的对话次数:{len(messages)}，总消耗的token数量:{total_token}")
    
