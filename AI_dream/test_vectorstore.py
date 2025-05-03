# test_vectorstore_with_score.py
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 初始化 embedding 模型
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# 设置路径
vectorstore_path = "faiss_psych_db"

# 安全加载数据库
if not os.path.exists(vectorstore_path):
    print(f"❌ Vector store `{vectorstore_path}` not found.")
    exit()

print(f"📦 Loading FAISS vector store from `{vectorstore_path}/`...")
db = FAISS.load_local(vectorstore_path, embedding, allow_dangerous_deserialization=True)

# 用户输入查询
query = input("Describe your dream: ").strip()
if not query:
    print(" Query is empty. Exiting.")
    exit()

# 相似度搜索 + 分数
results = db.similarity_search_with_score(query, k=3)

# 输出结果
print(f"\n🔍 Searching for: {query}")
print(f"Top {len(results)} results with similarity scores:\n")

if results:
    for i, (doc, score) in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Score: {score:.4f}")
        print(doc.page_content[:500])
        print("-" * 40)
else:
    print("⚠️ No results found.")
