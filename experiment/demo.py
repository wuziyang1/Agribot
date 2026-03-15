from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
# pip install rank_bm25
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


import os
from dotenv import load_dotenv

load_dotenv()


# 本地embedding模型地址
embedding_model_path = r'D:\LLM\Local_model\BAAI\bge-large-zh-v1___5'
# 初始化嵌入模型（用于文本向量化）
embeddings_model = HuggingFaceEmbeddings(
    model_name=embedding_model_path
)

# 加载文档
loader = TextLoader("deepseek介绍.txt", encoding="utf-8")
docs = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
)
split_docs = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=split_docs, embedding=embeddings_model
)
question = "社会影响"

# 向量检索
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
vector_retriever_doc = vector_retriever.invoke(question)
print("-------------------向量检索-------------------------")
print(vector_retriever_doc)

# 关键词检索 BM25Retriever是一种经典的文本检索算法，广泛应用于搜索引擎和文档匹配任务,核心是通过统计词项频率和文档结构特征，计算查询与文档的相关性得分
BM25_retriever = BM25Retriever.from_documents(split_docs, k=3)
BM25Retriever_doc = BM25_retriever.invoke(question)
print("-------------------BM25检索-------------------------")
print(BM25Retriever_doc)


# 混合检索
retriever = EnsembleRetriever(retrievers=[BM25_retriever, vector_retriever], weights=[0.5, 0.5])
retriever_doc = retriever.invoke(question)
print("-------------------混合检索-------------------------")
print(retriever_doc)

# 创建llm
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url")
)

# 创建prompt模板
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""

# 由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

# 创建chain
chain1 = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()
chain2 = RunnableMap({
    "context": lambda x: vector_retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

print("------------模型回复------------------------")
print("------------向量检索+BM25[0.5, 0.5]------------------------")
print(chain1.invoke({"question": question}))
print("------------向量检索------------------------")
print(chain2.invoke({"question": question}))
