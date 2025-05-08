from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os

load_dotenv()

def build_qa_chain():
    # Load Sri Lanka travel data
    loader = TextLoader("E:\Semester 8\Intelligent systems and Machine learning\ML_lecture_series\AI series\AI series\Travel_planner\data\sri_lanka.txt")
    docs = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Mistral Embeddings
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=os.getenv("MISTRAL_API_KEY"))
    
    # Vector Store (FAISS)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # Mistral Chat Model
    llm = ChatMistralAI(model="mistral-small", mistral_api_key=os.getenv("MISTRAL_API_KEY"))

    # Prompt Template
    prompt = ChatPromptTemplate.from_template("""
    You are a Sri Lanka travel expert. Answer the question based on the context below:

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # RAG Chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain