__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os 
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import streamlit as st

# ------ TITLE ----
st.set_page_config(page_title="Question Answering with RAG!", layout="wide")
st.markdown("""
    # Question Answering with RAG!

    Upload a PDF and input your OpenAI API key, then ask a question and press run.
""")

col1, col2 = st.columns(2)
with col1:
    file_input = st.file_uploader("Upload a PDF file")
with col2:
    openaikey = st.text_input("OpenAI API Key", placeholder="Enter your OpenAI API Key here...", type="password")
prompt = st.text_area("Enter your question below", height=50, placeholder="As defined by Prof. Nürnberger, what is...")
run_button = st.button("Run!")

# ------ Sidebar ------
st.sidebar.markdown("## Advanced Settings")
chunksize = st.sidebar.text_input("Chunk Size", placeholder="Default: 400", value=400, help="The number of characters or words in each segment")
chunkoverlap = st.sidebar.text_input("Chunk Overlap", placeholder="Default: 30", value=30, help="Portion of text shared between adjacent chunks")
select_k = st.sidebar.slider("Number of relevant chunks", min_value=1, max_value=5, step=1, value=2, help="Amount of 'documents' to return (Default: 2)")
select_chain_type = st.sidebar.radio(
    'Chain type', 
    options=['stuff', 'map_reduce', 'refine', 'map_rerank'],
    index=0, help="See more at https://docs.langflow.org/components/chains#combinedocschain."
)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

def qa(file, query, chain_type, k, chunksize, chunkoverlap):
    # Load document
    loader = PyPDFLoader(file)
    documents = loader.load()
    # Split the documents into chunks
    # text_splitter = CharacterTextSplitter(chunk_size=chunksize, chunk_overlap=chunkoverlap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunksize, chunk_overlap=chunkoverlap)
    texts = text_splitter.split_documents(documents)
    # Select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # Create the vectorstore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # Expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # Create a chain to answer questions 
    
    # Memory only stored if chain_type = stuff.
    
    if chain_type=="stuff":
        qa = ConversationalRetrievalChain.from_llm(llm=OpenAI(temperature=0.15),chain_type=chain_type, retriever=retriever, memory=memory,
            # combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT}, 
            )
        result = qa({"question": query})
    else:
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
        result = qa({"query": query})
    return result

@st.cache_data
def qa_result(file, query, chain_type, k, chunksize, chunkoverlap):
    os.environ["OPENAI_API_KEY"] = openaikey
    # Save pdf file to a temp file 
    if file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(file.read())
        
        if query:
            result = qa(file="temp.pdf", query=query, chain_type=chain_type, k=k, chunksize=chunksize, chunkoverlap=chunkoverlap)
            return result

if run_button:
    result = qa_result(file_input, prompt, select_chain_type, select_k, int(chunksize), int(chunkoverlap))
    if result:
        if select_chain_type=="stuff":
            st.write(result['answer'])
        else:
            st.write(result['result'])

