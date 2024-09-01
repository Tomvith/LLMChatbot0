# Importing necessary packages
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import TextLoader, CSVLoader  # Loaders for text and CSV documents
from PyPDF2 import PdfReader  # PyPDF2 is used to read PDF files
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveJsonSplitter
from langchain.schema import Document  # Document schema for text data
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
import textwrap
from langfuse import *
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
import sentence_transformers
import json
import yaml
import os
import tqdm
#import python_magic
from pathlib import Path
from datetime import datetime, date, time
import time
#import date
import sys
import tempfile  # Create temporary files

# Set the current woring directory / folder
sys.path.insert(0, os.path.abspath("."))

# Select the LLM / SLM / MLM 
llm = Ollama(model='llama3.1:latest') # "or any other model that you have"llama3.1 / phi3 wizardlm2 from Microsoft
#llm = Ollama(base_url="https://formally-flowing-whippet.ngrok-free.app/",model="llama3")

# set the embedding model path and name
embed_model = "all-MiniLM-L6-v2"

# set the dir under which the embeddings to be saved in a vector store
storing_path = "Json_FAISS_vectorstore"
storing_path_2 = "FAISS_vectorstore/PDF"

chat_history = []

# Load the input files from a dir, split the contents and convert to documents 
def load_split_doc(file_path):
    file_counter = 0
    for input_file in os.listdir(file_path):
        #print('input_file name ==', input_file)
        # Loading json files
        if input_file.endswith('.json'): 
            in_file = file_path + '/' + input_file
            with open(in_file, 'r', encoding='utf8') as f:
                json_data = json.load(f)
                print('json file name ==', in_file)
                # Initializing the RecursiveCharacterTextSplitter with max_chunk_size 
                json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
                # Splitting the input file content into chunks
                json_chunks = json_splitter.split_json(json_data=json_data)
                # Creating the documents from chunks
                json_docs = json_splitter.create_documents(texts=[json_chunks], convert_lists=True)
                #print("number of json docs = ",len(json_docs))
                file_counter += 1 

        if file_counter == 1:
            combined_json_docs = json_docs
        elif file_counter > 1:
            combined_json_docs = combined_json_docs + json_docs
        else:
            print("ERROR: No file loaded...")
            sys.exit(1)

    time.sleep(5)

    print("Total files loaded==", file_counter)
    print('len of combined_json_docs=',len(combined_json_docs))    
    
    # Returning the combined json documents
    return combined_json_docs

# Load the embedding model
def create_embedding_model(embed_model, normalize_embedding=True):
    #'''
    return HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={'device':'cpu'}, # here we will run the model with CPU only
        encode_kwargs = {
            'normalize_embeddings': normalize_embedding # keep True to compute cosine similarity
        }
    ) #'''
    #return OllamaEmbeddings(model = model_path)

# Create embeddings using FAISS and store them in local storing_path
def create_embeddings(documents, embed_model, storing_path):
    # Create the embeddings using FAISS. Comment the below line if you want to refer the pre-built vector store.
    vector_store = FAISS.from_documents(documents, embed_model)
    
    # Save the model in current directory. Comment the below line if you want to refer the pre-built vector store.
    vector_store.save_local(storing_path)
    
    # load the vector store from storing_path
    vector_store = FAISS.load_local(storing_path, embed_model, allow_dangerous_deserialization=True)
    
    # returning the vector_store
    return vector_store

# Set context based retriever chain with LLM model, vector store retriever and chat prompt
def get_context_retriever_chain(user_query,vector_store):
    # set up the llm, retriever and prompt to the retriever_chain
    # retriever_chain -> retrieve relevant information from the database

    #retriever = vector_store.as_retriever(search_type="similarity_score_threshold", k=2, search_kwargs={"score_threshold": 0.5}) #search_type='mmr', k=4) # To do: test `k`
    # search_type = similarity search (similarity) / maximum marginal relevance (mmr) search (a technique that helps retrieve documents that are both relevant to your query and diverse in their content)
    #fetch_k: Number of Documents to fetch before filtering to pass to MMR algorithm.
    # k - number of documents retrieved by the retriever. #lambda_mult-degree of diversity    
    #retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}) 
    print("\n**************inside get_context_retriever_chain to fetch the matching documents*************\n")
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5})
    
    # --> testing purpose only
    # Get k times relevant documents from vector store.
    docs = retriever.invoke(user_query)

    doc_count = 0
    for doc in docs:
        doc_count +=1
        print("Document #"+ str(doc_count) +": " +doc.page_content+"\n")
    
    # testing purpose only <--

    # Contextualize the question    
    context_q_system_prompt = """You are a helpful assistant for Standard Charterted Bank Open Banking API Portal. \
    You are smart enough to refer the local vector store for relevant documents for response. \
    The user will reward you if you provide exact answer for each user query. \
    Setup the response context based on the Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}") 
        ]
    )
    #("user", "Given the above conversation, generate a search query to look up in order to get the information relevant to the conversation")

    history_aware_retriever = create_history_aware_retriever(
        llm, 
        retriever, 
        context_q_prompt
    )

    return history_aware_retriever

# Get conversation type RAG chain which remembers the chat history
def get_conversational_rag_chain(history_aware_retriever):
    # Summarize the contents of the context obtained from the input json files and stored as embeddings in local vectorstore
    # Based on context, generate the answer of the question

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say politely and respectfully that you don't know. \
    Provide answer in detail and explain the same as and when needed.\ 
    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    qa_stuff_doc_chain = create_stuff_documents_chain(llm,qa_prompt)

    return create_retrieval_chain(history_aware_retriever, qa_stuff_doc_chain)


# Prettify the chatbot response
def get_response(user_query, vector_store):
    history_aware_retriever = get_context_retriever_chain(user_query,vector_store)
    conversational_rag_chain = get_conversational_rag_chain(history_aware_retriever)
            
    response = conversational_rag_chain.invoke({
        "input": user_query,
        "chat_history": chat_history
    })
    
    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(AIMessage(content=response['answer']))

    return response['answer']
######################################################################################################        
# function calls for pdf, text, csv and json files
# Read data from uploaded files
def read_data(files, loader_type):
    documents = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        try:
            if loader_type == "PDF":
                pdf_reader = PdfReader(tmp_file_path)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    documents.append(Document(page_content=text, metadata={"source": file.name, "page_number": page_num + 1}))
            elif loader_type == "Text":
                loader = TextLoader(tmp_file_path, encoding = 'UTF-8')
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file.name
                documents.extend(docs)
            elif loader_type == "JSON":
                loader = JSONLoader(tmp_file_path, jq_schema=".", text_content=False)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file.name 
                documents.extend(docs)
            elif loader_type == "CSV":
                loader = CSVLoader(tmp_file_path, encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file.name
                documents.extend(docs)
        finally:
            os.remove(tmp_file_path)

    return documents

# Split text into chunks
def get_chunks(texts, chunk_size, chunk_overlap):
    chunks = []
    if st.session_state.data_type in ['PDF','Text']:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for text in texts:
            split_texts = text_splitter.split_text(text.page_content)
            for split_text in split_texts:
                chunks.append(Document(page_content=split_text, metadata=text.metadata))

    elif st.session_state.data_type == "JSON":
        #json_splitter = RecursiveJsonSplitter(max_chunk_size=chunk_size)
        # Splitting the input file content into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for text in texts:
            split_texts = text_splitter.split_text(text.page_content)
            #split_texts = json_splitter.split_text(json_data=text.page_content)
            #split_texts = json_splitter.split_json(json_data=texts)
            for split_text in split_texts:
                chunks.append(Document(page_content=split_text, metadata=text.metadata)) 

    elif st.session_state.data_type == "CSV":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)   #, separators="\n")
        for text in texts:
            split_texts = text_splitter.split_text(text.page_content)
            for split_text in split_texts:
                chunks.append(Document(page_content=split_text, metadata=text.metadata)) 

    return chunks

# Store text chunks in a vector store using FAISS
def prep_vector_store(text_chunks, embed_model, vector_store_path):
    embeddings = create_embedding_model(embed_model) #OllamaEmbeddings(model=embedding_model_name)
    vector_store = FAISS.from_texts(texts=[doc.page_content for doc in text_chunks], embedding=embeddings, metadatas=[doc.metadata for doc in text_chunks])
    vector_store.save_local(vector_store_path)
    #if st.session_state.data_type in ['PDF','Text']:
    #else:
    #    vector_store = FAISS.from_documents(text_chunks, embeddings)

# Load the vector store using FAISS
def load_vector_store(embedding_model_name, vector_store_path):
    embeddings = create_embedding_model(embed_model) #OllamaEmbeddings(model=embedding_model_name)
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

######################################################################################################    

def main():

    # streamlit chatbot app config
    st.set_page_config(page_title="SCB Open Banking APIs ChatBot", page_icon="⚛️")
    st.title("Transformers AI Bot")

    with st.sidebar:
        st.title("Documents")        
        st.session_state.data_type = st.selectbox(
            "Select Data Type",
            ["Open Banking API","JSON","PDF", "Text", "CSV"]
        )
        if st.session_state.data_type not in ["Open Banking API"]:
            data_files = st.file_uploader("Upload your Files and Click on the Submit & Process Button", accept_multiple_files=True)

            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_documents = read_data(data_files, st.session_state.data_type)
                    text_chunks = get_chunks(raw_documents, chunk_size=2000, chunk_overlap=0)
                    prep_vector_store(text_chunks, embed_model, storing_path_2)
                    # Check if there are already info stored in the vectorDB
                    #if "vector_store" not in st.session_state:
                    st.session_state.vector_store = load_vector_store(embed_model,storing_path_2)
                    st.success("Done")

        else:
            # Logic For Open API JSON Files. Load the files, split them into chunks, convert to documents and store them in vector store
            # Load all the input files from the file_path
            #print("before load_split_doc = ", datetime.now())
            # load_split_doc() is commented to refer the pre-built vector store
            #documents = load_split_doc(file_path="data/JSON")
            #print("after load_split_doc = ", datetime.now())

            # Create Embedding Model
            embedding_model = create_embedding_model(embed_model) #"nomic-embed-text") #all-Minilm

            # create vectorstore. 
            #print("before create_embeddings = ", datetime.now())
            # create_embeddings() is commented to refer the pre-built vector store
            #st.session_state.vectorstore = create_embeddings(documents, embedding_model, storing_path)
            #print("After create_embeddings = ", datetime.now())
            
            # refer the current local FAISS vector store saved under the storing_path dir
            #if "vector_store" not in st.session_state:
            st.session_state.vector_store = FAISS.load_local(storing_path, embedding_model, allow_dangerous_deserialization=True)            

    ######################################################################################################    
    # Invoke the chains created to generate a response to a given user query
    #retriever_chain = get_context_retriever_chain(vector_store)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a SCB Open Banking API ChatBot. How can I help you?"),
        ]
    
    # user input
    user_query = st.chat_input("Type your query here...")
    if user_query is not None and user_query != "":
        start_time = datetime.now()
        # get conversational response for the user query
        response = get_response(user_query, st.session_state.vector_store)
        
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

        end_time = datetime.now()
        time_difference = (end_time - start_time).total_seconds()
        print("User query run time=" + str(time_difference) + "seconds")

    # conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Transformers"):
                st.write(message.content)

if __name__ == '__main__':
    main()

######################################################################################################    