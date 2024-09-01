# LLM Chatbot Project

* `open_banking_api_chatbot_v2.py` - This python program reads the pre-built FAISS vector store for SCB Open Banking APIs Chatbot and responds to user queries using LAAMA3.1:8b LLM. 
The app also supports selecting adhoc PDF, Text, CSV & JSON files, splitting, embedding and storing documents in FAISS vector store. Post that, user can query using the Chatbot.

* `prep_vector_store.py` - This python programs helps to 
    a. Load list of JSON files under a folder 
    b. Split file content using RecursiveJsonSplitter 
    c. Convert the chunks into documents and 
    d. Store them in the FAISS vector using the embedding model "all-MiniLM-L6-v2".

## Getting Started

1. Install Pipenv for Virtual Environments

    ```sh
    pip install --user pipenv
    ```

2. Start the virtual environment from the directory

    ```sh
    pipenv shell
    ```

3. Install requirements

    ```sh
    pip install -r requirements.txt
    ```

## LLM Model

You can either use the local Ollama server or one hosted in Google Colab. Python program will point to the Ollama running on localhost.

### Google Colab

**To run in Google Colab copy this folder at root of your Google Drive.**

You can now run all in one Google Colab Notebook <a href="https://colab.research.google.com/drive/1ZkatJVSV_qYyHK535A1DKVojGj9xz2lP?usp=sharing">LLM Chatbot Project</a>
