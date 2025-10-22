# Pinecone
# Cohere embedding
# Command Rplus api
#from langchain_cohere.llms import Cohere
import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env

COHERE_API_KEY = os.getenv("cohere_api_key")
PINECONE_API_KEY=os.getenv("pinecone_api_key")
from langchain_cohere import CohereEmbeddings,ChatCohere
import streamlit as st
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
#from langchain_groq import ChatGroq
import requests
proxies = {
    "http": None,
    "https": None,
}
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"

# function to initialize the pinecone database as a vector store for RAG pipeline
@st.cache_resource
def init_pinecone_retriever():
    
    embeddings = CohereEmbeddings(
    model="embed-english-v3.0",cohere_api_key=COHERE_API_KEY )
    pc=Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index('finance-documents-working-llamapaese-new')
    vectorstore=PineconeVectorStore(index=index, embedding=embeddings)
    retriever_pinecone=vectorstore.as_retriever(search_kwargs={"k":10})
    
    return retriever_pinecone   

    
# retrieve the top k documents from the vector store 
def retrieve_top_k_docs(retriever,user_question,top_k):
    
    # Retrieve the top 5 similar documents based on the user query
    retrieved_docs=retriever.vectorstore.similarity_search(user_question,top_k)
    
    return retrieved_docs


  
# construct prompt for cohere llm
def construct_prompt_cohere(user_question,retrieved_docs):
    
    system_prompt="""Use the following context as your learned knowledge, inside <context>{context}</context> XML tags.
    When answering the user:
    - If you don't know, just say that you don't know. Do not fetch ARC value when asked for MRC value.
    - If you are not sure, ask for clarification.
    - Avoid mentioning that you obtained the information from the context.
    - Given the context information, answer the query.
    - Always answer in a single sentence. Be brief and direct."""
    
    
    final_context=""
    
    for doc in retrieved_docs:
        final_context+=doc.page_content+"\n" 
    
    system_prompt=system_prompt.format(context=final_context)
        
    cohere_rag_prompt=[{"role":"system","content":system_prompt},
              {"role":"user","content":user_question}]
    
      
    return cohere_rag_prompt 
    


def generate_with_deepseek(cohere_rag_prompt):
    response = requests.post(

            VLLM_API_URL,proxies=proxies,

            headers={"Content-Type": "application/json"},

            json={

                "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",

                "messages": cohere_rag_prompt,

                "max_tokens": 512,

                "temperature": 0,

            },

        ).json()
    result=response['choices'][0]['message']['content']
    result=result.split('</think>')
    return result[1]
    