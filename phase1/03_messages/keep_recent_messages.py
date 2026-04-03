import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 加载环境变量
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 GROQ_API_KEY\n"
        "访问 https://console.groq.com/keys 获取免费密钥"
    )

# 初始化模型
model = init_chat_model("groq:llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

# 定义一个方法：保留最近两轮的对话记录
def keep_recent_messages(messages,max_pairs = 3):
    # 分离系统消息和用户消息
    system_messages = [m for m in messages if m.get("role") == "system"]
    user_messages = [m for m in messages if m.get("role") != "system"]

    # 只保留最近两轮的历史对话
    max_messages = max_pairs * 2
    recent_messages = user_messages[-max_messages:]

    # 返回最终保留的近两轮的对话记录以及系统消息
    return system_messages + recent_messages


# 模拟长对话
long_conversation = [
    {"role": "system", "content": "你是助手"},
    {"role": "user", "content": "你是啥模型"},
    {"role": "assistant", "content": "省略"},
    {"role": "user", "content": "你是哪个公司研发的模型？"},
    {"role": "assistant", "content": "省略"},
    {"role": "user", "content": "你都会些啥？"},
    {"role": "assistant", "content": "省略"},
    {"role": "user", "content": "你想干点啥？"},
    {"role": "assistant", "content": "无"},
    {"role": "user", "content": "第5个问题"},
]
print(f"原始消息数: {len(long_conversation)}")

# 优化：只保留最近 2 轮
optimized = keep_recent_messages(long_conversation, max_pairs=2)
print(f"优化后消息数: {len(optimized)}")
print(f"保留的内容: system + 最近2轮对话")

# 使用优化后的历史
response = model.invoke(optimized)
print(f"\nAI 回复: {response.content[:100]}...")

print("\n💡 技巧：对话太长时，只保留最近的几轮即可")