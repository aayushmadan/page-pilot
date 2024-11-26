# Chat with multiple pdfs

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_pdf_text(pdf_docs):
    text = "" # raw text
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/gtr-t5-base")
    embeddings = HuggingFaceInstructEmbeddings(model_name="intfloat/e5-base-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
    
def get_conversation_chain(vectorstore):
    llm = HuggingFaceEndpoint(
            # repo_id="google/flan-t5-xxl",
            # repo_id="tiiuae/falcon-7b-instruct",
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=0.5, 
            model_kwargs={
            "max_length": 512
            }
        )
    
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return conversation_chain
    
def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']
    
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
                
    # st.write(response)
    

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF-Pilot", page_icon="📑")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    st.header("PDF-Pilot")
    
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question: 
        handle_userinput(user_question)
    
    with st.sidebar:
        st.subheader("Your documents")
        
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
                
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
 
if __name__ == '__main__':
    main()