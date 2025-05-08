from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

def build_qa_chain():
    # Load and split the text file
    loader = TextLoader("data/sri_lanka.txt")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Use HuggingFace embeddings instead of OpenAI
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever()

    # Load GPT4All model
    llm = GPT4All(
        model="C:/Users/acer/gpt4all/resources/nomic-embed-text-v1.5.f16.gguf", 
        backend="gptj",                      # or "llama", depending on your model
        verbose=True
    )

    # Build the QA chain
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain


from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

def build_qa_chain():
    loader = TextLoader("data/sri_lanka.txt")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever()
    llm = OpenAI(temperature=0)

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain
