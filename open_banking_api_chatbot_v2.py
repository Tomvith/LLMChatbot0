# Importing necessary packages
import streamlit as st
from langchain_community.document_loaders import TextLoader, CSVLoader  # Loaders for text and CSV documents
from PyPDF2 import PdfReader  # PyPDF2 is used to read PDF files
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveJsonSplitter
from langchain.schema import Document  # Document schema for text data
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # Facebook AI Similarity Search
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage
from langfuse import *
from langchain_community.llms import Ollama
import sentence_transformers #for huggingface embedding models
import json
import os
from datetime import datetime, date, time
import time
import sys
import tempfile  # Create temporary files
#from langchain_community.document_loaders import DirectoryLoader
#from langchain_community.document_loaders import JSONLoader
#from langchain_community.document_loaders import PyMuPDFLoader
#from langchain_core.runnables.history import RunnableWithMessageHistory
#import textwrap
#import yaml
#import tqdm
#import python_magic
#import date
#from langchain.chains import RetrievalQA
#from langchain_community.chat_models import ChatOllama
#from langchain_core.prompts import PromptTemplate
#from langchain_community.embeddings import OllamaEmbeddings
#from pathlib import Path
######################################################################################################################################

# Set the current woring directory / folder
sys.path.insert(0, os.path.abspath("."))

# Select the LLM / SLM / MLM 
llm = Ollama(model="llama3.1:latest") #, temperature=0) # "or any other model that you have"llama3.1 / phi3 wizardlm2 from Microsoft

# set the embedding model path and name
embed_model = "all-MiniLM-L6-v2"

# set the dir under which the embeddings to be saved in a vector store
storing_path_list = ["Json_FAISS_vectorstore/500", "Json_FAISS_vectorstore/700", "Json_FAISS_vectorstore/1000", "Json_FAISS_vectorstore/1500"]   # for prebuilt json vector store
storing_path_main = ""
storing_path_adhoc = "FAISS_vectorstore/adhoc"       # for adhoc pdf, txt, csv and json files vector store


chat_history = []

# Load the input files from a dir, split the contents and convert to documents 
def load_split_doc(file_path):
    file_counter = 0
    for input_file in os.listdir(file_path):
        # Loading json files
        if input_file.endswith('.json'): 
            in_file = file_path + '/' + input_file

            with open(in_file, 'r', encoding='utf8') as f:
                json_data = json.load(f)
                print('json file name ==', in_file)
                title = json_data["info"]["title"]
                # Initializing the RecursiveCharacterTextSplitter with max_chunk_size 
                json_splitter = RecursiveJsonSplitter(max_chunk_size=st.session_state.maximum_chunk_size, min_chunk_size=st.session_state.minimum_chunk_size)
                # Splitting the input file content into chunks
                json_chunks = json_splitter.split_text(json_data=json_data)
                # Creating the documents from chunks
                json_docs = json_splitter.create_documents(texts=[json_chunks], convert_lists=True, metadatas=[{"source":title}])
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

# Create embeddings using FAISS and store them in local storing_path
def create_embeddings(documents, embed_model, storing_path_main):
    # Create the embeddings using FAISS. Comment the below line if you want to refer the pre-built vector store.
    vector_store = FAISS.from_documents(documents, embed_model)
    
    # Save the model in current directory. Comment the below line if you want to refer the pre-built vector store.
    vector_store.save_local(storing_path_main)
    
    # load the vector store from storing_path
    vector_store = FAISS.load_local(storing_path_main, embed_model, allow_dangerous_deserialization=True)
    
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
    print("\n**************inside get_context_retriever_chain to fetch the matching documents*************\n")

    if st.session_state.data_type == "Open Banking API": 
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k":5, "fetch_k":20, "lambda_mult": 0.5})
    else:
        retriever = vector_store.as_retriever(search_type=st.session_state.vector_search_type, search_kwargs={"k": st.session_state.k_value, "fetch_k": (st.session_state.k_value * 3), "lambda_mult": 0.5})
    # Get k times relevant documents from vector store.
    docs = retriever.invoke(user_query)

    # testing purpose only
    doc_count = 0
    for doc in docs:
        doc_count +=1
        print("Document #"+ str(doc_count) +": " +doc.page_content+"\n")
    
    # Contextualize the question    
    context_q_system_prompt = """You are a helpful assistant for Standard Chartered Bank Open Banking API Portal. \
    You are smart enough to refer the local vector store for relevant documents for response. \
    This vector store has documents in json fomrat. So you are expected to parse the json document content for accurate response.\
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

    qa_stuff_doc_chain = create_stuff_documents_chain(llm, qa_prompt)

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
    chunks = []
    file_counter = 0
    #print("inside read_data file names",files)
    #for file in files:
    for file in files:
        #in_file = files + '/' + input_file
        #print("file content", files)
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        try:
            if loader_type == "PDF":
                pdf_reader = PdfReader(tmp_file_path)   
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    documents.append(Document(page_content=text, metadata={"source": file.name, "page_number": page_num + 1}))
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state.chunk_size_val, chunk_overlap=st.session_state.chunk_overlap_val)
                for text in documents:
                    split_texts = text_splitter.split_text(text.page_content)
                    for split_text in split_texts:
                        chunks.append(Document(page_content=split_text, metadata=text.metadata))

                file_counter += 1

            elif loader_type == "Text":
                loader = TextLoader(tmp_file_path, encoding = 'UTF-8')
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file.name

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state.chunk_size_val, chunk_overlap=st.session_state.chunk_overlap_val)
                for text in docs:
                    split_texts = text_splitter.split_text(text.page_content)
                    for split_text in split_texts:
                        chunks.append(Document(page_content=split_text, metadata=text.metadata))

                file_counter += 1

            elif loader_type == "JSON":
                with open(tmp_file_path, 'r', encoding='utf8') as f:
                    docs = json.load(f)

                # get the title of the json file to save it as metadata in documents
                title = docs["info"]["title"]

                json_splitter = RecursiveJsonSplitter(max_chunk_size=st.session_state.maximum_chunk_size, min_chunk_size=st.session_state.minimum_chunk_size)                
                split_texts = json_splitter.split_text(json_data=docs)
                for split_text in split_texts:
                    chunks.append(Document(page_content=split_text, convert_lists=True ,metadata={"source":title}))

                file_counter += 1

            elif loader_type == "CSV":
                loader = CSVLoader(tmp_file_path) #, encoding="utf-8") has been removed due to load error
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file.name
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state.chunk_size_val, chunk_overlap=st.session_state.chunk_overlap_val)
                for text in docs:
                    split_texts = text_splitter.split_text(text.page_content)
                    for split_text in split_texts:
                        chunks.append(Document(page_content=split_text, metadata=text.metadata))  

                file_counter += 1              
        
            # combine the chunks from all input files
            if file_counter == 1:
                combined_docs = chunks
            elif file_counter > 1:
                combined_docs = combined_docs + chunks
            else:
                print("ERROR: No file loaded...")
                sys.exit(1)
            print("loaded file name", file.name)

        finally:
            os.remove(tmp_file_path)

    print("count of documents =", len(chunks))

    return combined_docs

# Store text chunks in a vector store using FAISS
def prep_vector_store(text_chunks, embed_model, vector_store_path):
    embeddings = create_embedding_model(embed_model) #OllamaEmbeddings(model=embedding_model_name)
    vector_store = FAISS.from_texts(texts=[doc.page_content for doc in text_chunks], embedding=embeddings, metadatas=[doc.metadata for doc in text_chunks])
    vector_store.save_local(vector_store_path)

# Load the vector store using FAISS
def load_vector_store(embed_model, vector_store_path):
    embeddings = create_embedding_model(embed_model) #OllamaEmbeddings(model=embedding_model_name)
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

    return vector_store

###################################################################################################################    

def main():
    # streamlit chatbot app config
    st.set_page_config(page_title="SCB Open Banking APIs ChatBot", page_icon="⚛️")
    st.title(":robot_face: :rainbow[Transformers AI Bot]")

    with st.sidebar:
        st.title(":bell: Selection Panel")
        st.markdown(":scissors: :blue[Set the below parameters for the selectecd Source/File Type]")
        st.session_state.data_type = st.selectbox("Select Source/File Type",["Open Banking API","JSON","PDF", "Text", "CSV","Generic"])

        if st.session_state.data_type not in ["Open Banking API","Generic"]:
            # Splitter parameters setting
            st.markdown(":scissors: :blue[_Splitter_ _Parameters_]")
            if st.session_state.data_type == "JSON":

                st.session_state.minimum_chunk_size = st.slider("Minimum Chunk Size", value=500, step=100, 
                                                                      min_value=100, max_value=5000, help="Recommended Min chunk size is 500.") #, placeholder="Type a number less than max chunk size")
                st.session_state.maximum_chunk_size = st.slider("Maximum Chunk Size", value=700, step=100, 
                                                                      min_value=100, max_value=5000, help="Recommended Max chunk size is 700.") #, placeholder="Type a number greater than min chunk size")                
            else:
                st.session_state.chunk_size_val = st.slider("Chunk Size", value=500, step=100, 
                                                                  min_value=100, max_value=5000, help="Recommended chunk size is 500-700.")
                st.session_state.chunk_overlap_val = st.slider("Chunk Overlap", value=0, step=5, 
                                                                     min_value=0, max_value=100, help="Recommended chunk overlap for Text file is 10-20. Rest of the files is 0.")
            # Retriever parameters setting
            st.markdown(":chains: :blue[_Retriever_ _Parameters_]")
            
            st.session_state.vector_search_type = st.selectbox("Select Vectors Search Type", ["Maximum Marginal Relevance","Similariy"]
                                                               , help="Recommended Maximum Marginal Relevance (MMR) if the Embedding model supports.")
            if st.session_state.vector_search_type == "Maximum Marginal Relevance":
                st.session_state.vector_search_type = "mmr"
            else:
                st.session_state.vector_search_type = "similarity"
            
            st.session_state.k_value = st.number_input("Number of similar Docs to be retrieved", value=5, step=1, min_value=1, max_value=50, 
                                                       placeholder="Type K value", help="Recommended value is 4-6.")
            
            # Uploald file(s)
            data_files = st.file_uploader("Upload your Files and Click Process Button", accept_multiple_files=True,)

            if st.button("Process Files"):
                with st.spinner("Processing..."):
                    text_chunks = read_data(data_files, st.session_state.data_type)
                    prep_vector_store(text_chunks, embed_model, "/content/drive/MyDrive/LLM_Chatbot/" + storing_path_adhoc)
                    st.session_state.vector_store = load_vector_store(embed_model,"/content/drive/MyDrive/LLM_Chatbot/" + storing_path_adhoc)
                    st.success("Done")

        elif st.session_state.data_type == "Open Banking API":
            # Create Embedding Model
            embedding_model = create_embedding_model(embed_model) #"nomic-embed-text") #all-Minilm

            # creating vectorstore. 
            # DO NOT DELETE. create_embeddings() is commented to refer the pre-built vector store
            #st.session_state.vectorstore = create_embeddings(documents, embedding_model, storing_path)

            # refer the current local FAISS vector store saved under the storing_path dir
            st.session_state.vector_search_type = st.selectbox("Select Vector Store with", ["chunk_size_700","chunk_size_500","chunk_size_1000","chunk_size_1500"], 
                                                               help="Recommended chunk size is 700")
            if st.session_state.vector_search_type == "chunk_size_700":
                storing_path_main = "/content/drive/MyDrive/LLM_Chatbot/" + storing_path_list[1]
            elif st.session_state.vector_search_type == "chunk_size_500":
                storing_path_main = "/content/drive/MyDrive/LLM_Chatbot/" + storing_path_list[0]
            elif st.session_state.vector_search_type == "chunk_size_1000":
                storing_path_main = "/content/drive/MyDrive/LLM_Chatbot/" + storing_path_list[2]                
            elif st.session_state.vector_search_type == "chunk_size_1500":
                storing_path_main = "/content/drive/MyDrive/LLM_Chatbot/" + storing_path_list[3]

            st.markdown(":point_right: :blue[Note: This 'Open Banking API' option will use the pre-built FAISS vector store to the fetch relevant documents for the user query.]")
            
            # point to the locally pre-built vectore store
            st.session_state.vector_store = FAISS.load_local(storing_path_main, embedding_model, allow_dangerous_deserialization=True)
        
        elif st.session_state.data_type == "Generic":
            st.markdown(":point_right: :blue[Note: Generic option will not use any vector store. It will direct the user query to the LLAMA3.1:8b LLM model and respond.]")

    ######################################################################################################    
    # Invoke the chains created to generate a response to a given user query

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a SCB ChatBot. How can I help you?"),
        ]
    
    time_difference = ''

    # user input
    user_query = st.chat_input("Type your query here...")
    if user_query is not None and user_query != "":
        start_time = datetime.now()
        
        # get conversational response for the user query
        # for Generic type, vector store will not be used. LLM will execute the query and provide response.
        if st.session_state.data_type == "Generic": 
            response = llm.invoke(user_query)
        else:
            response = get_response(user_query, st.session_state.vector_store)
        
        end_time = datetime.now()
        time_difference = int((end_time - start_time).total_seconds())

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("User"):
                st.write(message.content)
                query_time = "Time taken by the Bot to respond to this query: " + str(time_difference) + " seconds"
                st.write(query_time)                 

if __name__ == '__main__':
    main()

######################################################################################################    