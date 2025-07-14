import json
import os # os is used to interact with the operating system
import sys #system is used to add the current directory to the Python path
import boto3 # boto3 is used to interact with AWS services
import streamlit as st # streamlit is used to build web applications
import langchain_community # langchain_community is used for community contributions to LangChain

## We will be suing Titan Embeddings Model To generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter # used to split text into chunks
from langchain_community.document_loaders import PyPDFDirectoryLoader # used to load PDF files from a directory

# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS # FAISS is used for efficient similarity search in large datasets

## LLm Models
from langchain.prompts import PromptTemplate # used to create prompts for language models
from langchain.chains import RetrievalQA # used to create a question-answering chain

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)
#client is a method provided by boto3 that:Creates a low-level Python interface to communicate directly with a specific AWS service


## Data ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    #Each chunk will be at most 10,000 characters long.
    #This is useful to avoid hitting LLaMA 3 or Claude token limits
    #overlap helps models retain context between chunks when queried or embedded separately.
    
    docs=text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents( # FAISS is used for efficient similarity search in large datasets
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index") # Saves the vector store to a local directory

def get_claude_llm():
    ## Create the Anthropic Claude Model
    llm = Bedrock(
        model_id="anthropic.claude-v2",
        client=bedrock
    )
    return llm

def get_llama3_llm():
    ## Create the Meta LLaMA 3 Model
    llm = Bedrock(
        model_id="meta.llama3-8b-instruct-v1:0",
        client=bedrock
   
    )
    return llm


prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but atleast summarize with 
50 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    #RetrievalQA is a chain that combines a retriever and an LLM to answer questions based on retrieved documents
    #It retrieves relevant documents from the vector store and then uses the LLM to generate an answer based on those documents.
    #The retriever is set to use similarity search with k=3, meaning it will retrieve the top 3 most similar documents to the query.
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # "stuff" chain type is used to directly pass the retrieved documents to the LLM
    # The "stuff" chain type is suitable for smaller documents or when you want to pass
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT} # This sets the prompt template for the chain
)
    answer=qa({"query":query}) # This method takes a query and returns an answer based on the retrieved documents.
    return answer['result']


def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock's LLaMA 3 and Claude Models")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_claude_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_llama3_llm()

            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
