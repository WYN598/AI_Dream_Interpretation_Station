import os
import nltk
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# 环境加载
load_dotenv()

# -------------------------------
# 安全分句器（解决 punkt_tab 问题）
# -------------------------------
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

punkt_param = PunktParameters()
tokenizer = PunktSentenceTokenizer(punkt_param)

def safe_sent_tokenize(text):
    try:
        return tokenizer.tokenize(text)
    except:
        return text.split('.')  # fallback

# -------------------------------
# 语义 + 回退式分块函数
# -------------------------------
def hybrid_semantic_chunking(docs, sentences_per_chunk=3, overlap=1, max_chunk_chars=1000, min_words=10):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_chars,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    for doc in docs:
        try:
            sentences = safe_sent_tokenize(doc.page_content)
            for i in range(0, len(sentences), sentences_per_chunk - overlap):
                chunk_text = " ".join(sentences[i:i + sentences_per_chunk]).replace("\n", " ").strip()
                if len(chunk_text.split()) >= min_words:
                    if len(chunk_text) > max_chunk_chars:
                        sub_chunks = splitter.split_text(chunk_text)
                        for sub in sub_chunks:
                            chunks.append(Document(page_content=sub, metadata=doc.metadata))
                    else:
                        chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))
        except Exception as e:
            print(f" Failed to chunk document: {e}")
    return chunks

# -------------------------------
# 文档加载
print("🔍 Loading documents from `psych_docs/`...")
docs = []
for filename in os.listdir("psych_docs"):
    try:
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join("psych_docs", filename), encoding="utf-8")
            docs.extend(loader.load())
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("psych_docs", filename))
            docs.extend(loader.load())
    except Exception as e:
        print(f" Error loading {filename}: {e}")

print(f"Loaded {len(docs)} documents.")

# -------------------------------
#  分块
# -------------------------------
print("Performing hybrid semantic chunking...")
split_docs = hybrid_semantic_chunking(docs)

if not split_docs:
    print(" No valid chunks to embed. Exiting.")
    exit()

print(f" Generated {len(split_docs)} semantic chunks.")

# -------------------------------
# 构建向量库
# -------------------------------
print("Generating embeddings and building FAISS vector store...")
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
faiss_db = FAISS.from_documents(split_docs, embedding)

# -------------------------------
# 保存向量数据库
# -------------------------------
faiss_db.save_local("faiss_psych_db")
print("🎉 FAISS 向量数据库构建成功并已保存！")
