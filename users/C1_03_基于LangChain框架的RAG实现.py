# 初始化设置
import os
from dotenv import load_dotenv # python-dotenv库，把配置项从.env文件中读取除了，写入进程的环境变量os.environ，避免硬编码密钥、密码等敏感信息
from langchain_community.document_loaders import TextLoader # 把文件读成Document对象
from langchain_text_splitters import RecursiveCharacterTextSplitter # 把长文本切成小块

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

# 加载环境变量
load_dotenv()
# 加载文档
markdown_path = "../data/C1/markdown/easy-rl-chapter1.md"
loader = TextLoader(markdown_path) # 创建一个加载器实例
docs = loader.load() # 调用实例方法，执行读取，返回文档对象列表
# 文本分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=[
        "\n\n",           # 优先按照段落分割
        "\n",             # 其次按照换行分割
        "。",             # 中文句号
        "！",             # 中文感叹号
        "？",             # 中文问号
        "；",             # 中文分号
        "，",             # 中文逗号
        "、",             # 中文顿号
        ".",              # 英文句号
        ",",              # 英文逗号
        " ",              # 空格
        ""                # 最后实在无法分割时按字符分割
    ]

    ) # 创建分割器工具 ---- 配置
texts = text_splitter.split_documents(docs) # 执行 ---- 执行
# 索引构建
embedding = HuggingFaceEmbeddings( # 初始化中文嵌入模型
    model_name = "BAAI/bge-small-zh-v1.5",
    model_kwargs = {"device":"cpu"},
    encode_kwargs = {"normalize_embeddings":True}
)
# 构建向量存储
vectorstore = InMemoryVectorStore(embedding) # 创建一个内存向量库实例
vectorstore.add_documents(texts)                # 把分块后的文档批量加入向量库

# 查询与检索
question = "文中举了哪些例子？"
retrieved_docs = vectorstore.similarity_search(question, k=8)
# 准备上下文
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs) # 将检索到的多个文本块用“\n\n”连接起来

# 生成集成
# 构建提示词模板
prompt = ChatPromptTemplate.from_template("""
请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题”
上下文：
{context}

问题：
{question}

回答：

""")
# 配置大语言模型
llm = ChatDeepSeek(
    model = "deepseek-flash",
    temperature=0.7,
    max_tokens=2048,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)
# 调用LLM生成答案并输出
answer = llm.invoke(prompt.format(question=question,context=docs_content))
print(answer)
print(answer.content)






