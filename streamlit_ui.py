# this is a streamlit UI template


#[CHANGE THE IMPORT PIPELINE] to:

# PIPELINE 1: rag_pipeline_23ai_commandr
# PIPELINE 2: rag_pipeline_23ai_hf_llama
# PIPELINE 3: rag_pipeline_pinecone_commandr
# PIPELINE 4: rag_pipeline_pinecone_hf_llama

from rag_pipeline import init_pinecone_retriever,retrieve_top_k_docs,construct_prompt_cohere,generate_with_deepseek # [CHANGE THE IMPORT PIPELINE LIBRARIES & FUNCTION] rag_pipeline_23ai_commandr OR rag_pipeline_23ai_hf_llama OR rag_pipeline_pinecone_commandr OR rag_pipeline_pinecone_hf_llama
import streamlit as st
import time



retriever=init_pinecone_retriever()#[CHANGE THE INIT FUNCTION] use init_oracle_23ai_retriever() or init_pinecone_retriever() 


if __name__ == "__main__":

    st.header("Pi Chatbot")

        
    # [DONT CHANGE] check if the messages list is empty. If empty, initialize the messages list
    if "messages" not in st.session_state:
                st.session_state.messages = []
    else:
    # [DONT CHANGE] If messages list is not empty, display the messages in the chat window
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
    
    # [DONT CHANGE] Add a chat input to get the user query
    user_question = st.chat_input(placeholder="Enter your question....")
    
    # [DONT CHANGE] Check if the user has entered a query.
    if user_question != None:
        
        # [DONT CHANGE] Display the user query in the chat window
        with st.chat_message("user"):
            st.markdown(user_question)
        
        #  [DONT CHANGE] Append the user query to the session state messages list
        st.session_state.messages.append({"role": "user", "content": user_question})

        # [DONT CHANGE] Check if the last message in the messages list is from the user. 
        # If yes, it is AI's turn. So, retrieve the context and generate the model response.
        if st.session_state.messages[-1]['role']=='user':
            
                   
            
            retrieved_docs=retrieve_top_k_docs(retriever=retriever,user_question=user_question,top_k=5) #[CHANGE THE TOPK IF YOU WANT]
            
           
            rag_prompt=construct_prompt_cohere(user_question=user_question,retrieved_docs=retrieved_docs)  # [CHANGE FUNCTION] construct_prompt_cohere(retrieved_docs,user_question) or construct_prompt_llama_3_1(retrieved_docs,user_question)
            #model_id= [UNCOMMENT for gpu]
            #cache_dir= [UNCOMMENT for gpu]
            #model,tokenizer=load_model_to_gpu(model_id,cache_dir,max_new_tokens=100,temperature=0.1) [UNCOMMENT TO LOAD MODEL TO GPU]
            
            model_response=generate_with_deepseek(rag_prompt) # [CHANGE THE GENERATE RESPONSE FUNCTION]  generate_response_cohere(rag_prompt) or generate_response_llama_3_1(model,tokenizer,llama_3_1_rag_prompt,max_new_tokens=100,temperature=0.1)
            print(model_response)
            
           
            
            #[DONT CHANGE]
            with st.chat_message("ai"):
                st.write(model_response)
                

            
            #[DONT CHANGE]            
            # Append the model response to the session state messages list
            st.session_state.messages.append({'role':'ai','content':model_response})