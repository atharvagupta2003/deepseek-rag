from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

urls = [
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
]

loaded_docs = [WebBaseLoader(url).load() for url in urls]
docs = [item for sublist in loaded_docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=600, chunk_overlap=100
)
chunked_docs = text_splitter.split_documents(docs)

# Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=chunked_docs,
    collection_name="rag-deepseek",
    embedding=embeddings,
    persist_directory=os.environ.get("CHROMADB_PERSIST_DIR")
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})