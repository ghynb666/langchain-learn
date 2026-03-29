from itertools import product
from tempfile import template

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os

from pyexpat import features

load_dotenv()

model = ChatGroq(
    model="qwen/qwen3-32b",  # Groq 上的千问模型
    groq_api_key=os.getenv("GROQ_API_KEY")
)


# 方式2：显式的指定变量
template = PromptTemplate(input_variables=["product", "feature"],
                                 template="为{product}写一段广告语,重点突出{feature}特点")

# 向提示词模板传入参数
prompt = template.format(product="智能手表", feature="超长续航")
print(f"\n输入 : {prompt}")

response = model.invoke(prompt)
print(f"\nAI输出:{response.content}")

# 方式 3 :部分变量预填充
template1 = PromptTemplate.from_template("你是一个{role},请{task}")

# 对模板中的变量进行预填充
partial_template = template1.partial(role="Python 导师")

# 随后通过 format 填充
prompt1 = partial_template.format(task="解释装饰器")
print(f"\n输入 : {prompt1}")
response1 = model.invoke(prompt1)
print(f"\n输出 : {response1}")


# 方式 4 使用 invoke 方式直接传入模板参数
prompt2 = partial_template.invoke({"task": "解释迭代器"})
print(f"\n使用 invoke 直接传入：{prompt2}")
