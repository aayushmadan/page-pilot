# Chat with website

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv() 

def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create a vector store from chunks
    
    # vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    
    vector_store = Chroma.from_documents(document_chunks, HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl"))
    
    return vector_store

def get_context_retriever_chain(vector_store):
    # llm = ChatOpenAI()
    llm = HuggingFaceEndpoint(
            # repo_id="google/flan-t5-xxl",
            repo_id="tiiuae/falcon-7b-instruct",
            temperature=0.5, 
            model_kwargs={
            "max_length": 512
            }
        )
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
        ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain 

def get_conversational_rag_chain(retriever_chain):
    
    # llm = ChatOpenAI()
    llm = HuggingFaceEndpoint(
            # repo_id="google/flan-t5-xxl",
            repo_id="tiiuae/falcon-7b-instruct",
            temperature=0.5, 
            model_kwargs={
            "max_length": 512
            }
        )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system","Answer the user's questions based on the below {context}:\n\n"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    # create conversation chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)  
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    return response['answer']
    
# App config
st.set_page_config(page_title="PagePilot", page_icon="🤖")
st.title("PagePilot")
    
# Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Enter the URL")
    
else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        AIMessage(content="Hello! I am a bot, How can I help you")

        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
    
    
    # with st.sidebar:
    #     st.write(documents)
        
    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=user_query))
        
        # retrieved_documents = retriever_chain.invoke({
        #     "chat_history": st.session_state.chat_history,
        #     "input": user_query
        # })
        # st.write(retrieved_documents)
        
    # with st.sidebar:
    #     st.write(st.session_state.chat_history)

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
             
    
    