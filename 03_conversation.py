import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from groq import RateLimitError

load_dotenv()
model = ChatGroq(model="qwen/qwen3-32b", groq_api_key=os.getenv("GROQ_API_KEY"))

# 添加带重试机制的调用函数
def invoke_with_retry(model, messages, max_retries=3, base_wait=6):
    """调用模型时自动重试，处理速率限制错误"""
    for attempt in range(max_retries):
        try:
            return model.invoke(messages)
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = base_wait * (attempt + 1)
                print(f"⚠️  遇到速率限制，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise Exception(f"达到最大重试次数，请求失败：{str(e)}")

print("=" * 70)
print("练习 3：多轮对话实践")
print("=" * 70)

# 初始化对话
conversation = [
    {"role" : "system","content" : "你是一位友善的Python专家"}
]
print(f"\n第一轮对话\n")
# 第一轮对话
conversation.append({"role": "user", "content": "我打算学习python，我要从哪里开始?"})
print(f"\n第二轮对话输入:{conversation[-1]['content']}\n")

response1 = invoke_with_retry(model, conversation)

print(f"\n第一轮对话结果是:{response1.content}\n")

# 将对话内容保留至对话上下文中
conversation.append(
    {"role" : "assistant","content" : response1.content}
)
# 第二轮对话
print(f"\n第二轮对话\n")
conversation.append(
    {"role" : "user","content" : "python中的数据类型有哪些？"}
)
print(f"\n第二轮对话输入:{conversation[-1]['content']}\n")

response2 = invoke_with_retry(model, conversation)
print(f"\n第二轮对话结果:{response2.content}\n")

# 添加至对话上下文
conversation.append(
    {"role" : "assistant","content": response2.content}
)

# 第三轮对话
print(f"开启第三轮对话\n")
conversation.append(
    {"role" : "user", "content" : "我提问的第一个问题是啥?"}
)
print(f"第三轮输入的问题:{conversation[-1]['content']}\n")

response3 = invoke_with_retry(model, conversation)
print(f"\n第三轮输出:{response3.content}\n")
