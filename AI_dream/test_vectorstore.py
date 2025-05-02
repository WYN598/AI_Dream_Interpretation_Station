# test_vectorstore.py
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embedding = OpenAIEmbeddings()

# ✅ 加上允许反序列化（你自己构建的数据库是可信的）
db = FAISS.load_local("faiss_psych_db", embedding, allow_dangerous_deserialization=True)

query = "a dream about flying and seeing stars"
results = db.similarity_search(query, k=3)

print("🔍 Top 3 matching chunks:")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---\n{doc.page_content[:500]}")
