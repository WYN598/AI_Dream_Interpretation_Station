import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader

from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()  # 加载 OPENAI_API_KEY

# 1. 加载文档
docs = []
for filename in os.listdir("psych_docs"):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join("psych_docs", filename), encoding="utf-8")
        docs.extend(loader.load())
    elif filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join("psych_docs", filename))
        docs.extend(loader.load())

# 2. 切分文本

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
split_docs = splitter.split_documents(docs)

# 3. 生成嵌入向量并构建 FAISS 数据库
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
faiss_db = FAISS.from_documents(split_docs, embedding)

# 4. 保存数据库到本地
faiss_db.save_local("faiss_psych_db")

print("FAISS 向量数据库构建成功并已保存！")
