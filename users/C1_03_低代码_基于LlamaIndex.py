import os 
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# 全局设置LLM大模型
Settings.llm = OpenAILike(
        model = "deepseek-flash",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base="https://api.deepseek.com",
        is_chat_model=True
)

# 全局设置Embedding嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 读取指定的markdown文档
documents = SimpleDirectoryReader(input_files=["../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 基于加载的文档构建向量索引
index = VectorStoreIndex.from_documents(documents)

# 从向量索引构建查询引擎，用来实现RAG问答
query_engine = index.as_query_engine(similarity_top_k=8)

print(query_engine.get_prompts())

# 发起RAG问答，打印回复结果
print(query_engine.query("文中列举了哪些例子？"))