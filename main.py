import os

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

load_dotenv()

if __name__ == "__main__":
    
    pdf_path = "./pdf_to_read/bitcoin.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    documents = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index_pdf")

    new_vectorstore = FAISS.load_local("faiss_index_pdf", embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever()
    )
    
    response = qa.run("Give me the gist of Bitcoin in 3 sentences")
    
    print(response)
