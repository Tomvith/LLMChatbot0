# Importing necessary packages
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveJsonSplitter
from langchain_community.vectorstores import FAISS
from langfuse import *
from langchain_community.llms import Ollama
import sentence_transformers
import json
import os
from datetime import datetime, date, time
import time
import sys

# number of documents created for 46 JSON files with max_chunk_size 1500: 1367
# number of documents created for 46 JSON files with max_chunk_size 1000: 2083
# number of documents created for 46 JSON files with max_chunk_size  700: 3047
# number of documents created for 46 JSON files with max_chunk_size  500: 4633

storing_path = "Json_FAISS_vectorstore/cs_test"
file_path = "data/JSON"
min_chunk_val = 700
max_chunk_val = 700

# Load the input files from a dir, split the contents and convert to documents 
def load_split_doc(file_path):
    file_counter = 0
    for input_file in os.listdir(file_path):
        # Loading json files
        if input_file.endswith('.json'): 
            in_file = file_path + '/' + input_file
            with open(in_file, 'r', encoding='utf8') as f:
                json_data = json.load(f)
                
                # get title from input json file 
                title = json_data["info"]["title"]
                print("file title == ", title)

                # Initiale the RecursiveCharacterTextSplitter with max_chunk_size 
                json_splitter = RecursiveJsonSplitter(min_chunk_size=min_chunk_val, max_chunk_size=max_chunk_val)

                # Splitting the input file content into chunks
                json_chunks = json_splitter.split_text(json_data=json_data)
        
                # Create the documents from chunks
                json_docs = json_splitter.create_documents(texts=[json_chunks], convert_lists=True, metadatas=[{"source":title}])
                file_counter += 1 

        if file_counter == 1:
            combined_json_docs = json_docs
            for doc in combined_json_docs:
                print("\n\n doc list ===", doc)
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

# Loading the embedding model
def load_embedding_model(model_path, normalize_embedding=True):
    
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cpu'}, # here we will run the model with CPU only
        encode_kwargs = {
            'normalize_embeddings': normalize_embedding # keep True to compute cosine similarity
        }
    ) 

# Create embeddings using FAISS and store them in local storing_path
def create_embeddings(documents, embedding_model, storing_path=storing_path):
    # Creating the embeddings using FAISS - Facebook AI Similarity Search
    vector_store = FAISS.from_documents(documents, embedding_model)
    
    # Saving the model in current directory
    vector_store.save_local(storing_path)
    
    # load the vector_store from the local store path
    vector_store = FAISS.load_local(storing_path, embedding_model, allow_dangerous_deserialization=True)
    
    # returning the vectorstore
    return vector_store

######################################################################################################        

# Load all the input files from the file_path
documents = load_split_doc(file_path=file_path)

# Load the Embedding Model
embedding_model = load_embedding_model(model_path="all-MiniLM-L6-v2") 

start_time = datetime.now()
# creating vector_store
vector_store = create_embeddings(documents, embedding_model)

end_time = datetime.now()
time_difference = int((end_time - start_time).total_seconds())
print("Time taken to load the vector store = " + str(time_difference) + " seconds")
