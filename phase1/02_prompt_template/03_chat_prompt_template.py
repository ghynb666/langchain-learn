from itertools import product


from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain_groq import ChatGroq
import os

from pyexpat import features

load_dotenv()

model = ChatGroq(
    model="qwen/qwen3-32b",  # Groq 上的千问模型
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 元组的方式创建聊天提示词模板
template = ChatPromptTemplate.from_messages([("system", "你是一位{role}"), ("user", "{question}")])

# 给元组中的模板参数传入真实值
messages1 = template.format_messages(role="Python导师", question="什么是装饰器?")

# 传给model
response1 = model.invoke(messages1)
print(f"AI响应 ：{response1.content}")

# 在元组的基础上使用字符串简写
template2 = ChatPromptTemplate.from_messages([("system", "你是助手"), ("user", "{user_input}")])

template2.invoke({
    "user_input" : "你好"
})
# 高级用法，MessagePromptTemplate 类（高级，细粒度控制）
system_template = SystemMessagePromptTemplate.from_template("你是一位{role},你的特长是{specialty}")

human_template = HumanMessagePromptTemplate.from_template("关于{topic}，我想知道{question}")

chat_message = ChatPromptTemplate.from_messages([system_template, human_template])

# 使用
chat_message.format_messages(
    role="Python 专家",
    specialty="解决 Python 相关的问题",
    topic="Python 语言基础",
    question="Python 的基本数据类型有哪些？"
)