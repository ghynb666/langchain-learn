from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os

load_dotenv()

model = ChatGroq(
    model="qwen/qwen3-32b",  # Groq 上的千问模型
    groq_api_key=os.getenv("GROQ_API_KEY")
)

topic = "python"
difficulty = "初学者"
# 创建可复用的提示词模板
template = PromptTemplate.from_template(f"你是一位{difficulty}编程的导师，请用简单易懂的文本介绍{topic}")

# 输出整个定义的模板

print(f"\n模板 : {template.template}\n")
print(f"\n模板中的变量: {template.input_variables}\n")

# 将上述定义的提示词模板传入实际参数
prompt = template.format(difficulty=difficulty, topic=topic)
print(f"输入的提示词:{prompt}")

# 模型去调用该提示词
response = model.invoke(prompt)
print(f"\n响应结果:{response.content}\n")
