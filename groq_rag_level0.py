import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

TOKEN_FILE = os.path.expanduser('~') + '/groq_token_file.txt'

def read_token():
    with open(TOKEN_FILE, "r") as tf:
        return tf.readline()
    return ''

def run_it():
    # Step 1: Load the document from a web url
    loader = WebBaseLoader(["https://igorpak.wordpress.com/2020/12/10/what-if-they-are-all-wrong/"])
    documents = loader.load()

    # Step 2: Split the document into chunks with a specified chunk size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_splits = text_splitter.split_documents(documents)

    # Step 3: Store the document into a vector store with a specific embedding model
    vectorstore = FAISS.from_documents(all_splits, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

    # Query against your own data
    chain = ConversationalRetrievalChain.from_llm(llm,
                                                vectorstore.as_retriever(),
                                                return_source_documents=True)

    # no chat history passed
    result = chain({"question": "What is a conjecture?", "chat_history": []})
    print('Answer 1: \n' + result['answer'] + '\n---------\n')

    # This time your previous question and answer will be included as a chat history which will enable the ability
    # to ask follow up questions.
    query = "How do we prove it is impossible to disprove a conjecture?"
    chat_history = [(query, result["answer"])]
    result = chain({"question": query, "chat_history": chat_history})
    print('Answer 2: \n' + result['answer'] + '\n---------\n')

if __name__ == '__main__':
    os.environ["GROQ_API_KEY"] = read_token()
    run_it()