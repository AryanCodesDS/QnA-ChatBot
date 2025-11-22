import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.rag_utils import (
    SentenceTransformerEmbeddings, 
    generate_text, 
    rag_conversation, 
    get_session_history
)
from src.pdf_utils import load_and_process_pdfs

# Load environment variables
load_dotenv()

# Title
st.title("🤖 Question Answering Using LLMs")

# Initialize Embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed = SentenceTransformerEmbeddings(model_name)

list_models = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("Enter Groq API Key", type="password")
    model = st.selectbox("Select A Model", list_models)
    temperature = st.slider("Select Temperature", 0.0, 1.0, 0.5)
    max_tokens = st.slider("Select Max Tokens", 0, 1000, 100)
    st.divider()
    chat_with_pdf_option = st.checkbox("Chat with PDF")
    
    pdf_file = None
    session_id = "default_session"
    
    if chat_with_pdf_option:
        st.subheader("📂 Upload PDF")
        pdf_file = st.file_uploader(
            "Upload PDF", type=["pdf"], accept_multiple_files=True)
        session_id = st.text_input(
            "Enter Session ID", value="default_session")

# Main Logic
if chat_with_pdf_option:
    st.subheader("📄 Chat with PDF")
    
    if pdf_file:
        if 'vector_store' not in st.session_state or st.sidebar.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                documents = load_and_process_pdfs(pdf_file)
                st.session_state.vector_store = FAISS.from_documents(documents, embedding=embed)
                st.success("PDF Uploaded and Processed Successfully!")
        
        if 'vector_store' in st.session_state:
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            
            template = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", template),
                    MessagesPlaceholder("chat_history"),
                    ("user", "{input}"),
                ]
            )
            
            # Chat Interface
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if question := st.chat_input("Ask a question about your PDF..."):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            answer = rag_conversation(
                                prompt, question, model, api_key, temperature, max_tokens, retriever, session_id
                            )
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload a PDF file to start chatting.")

else:
    st.subheader("💬 General Q&A")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful question answering AI."),
            ("user", "{question}"),
        ]
    )

    st.markdown("This is a simple question answering application using LLMs. You can ask any question and the AI will try to answer it.")
    
    # Chat Interface for General Q&A
    if "gen_messages" not in st.session_state:
        st.session_state.gen_messages = []

    for message in st.session_state.gen_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask anything..."):
        st.session_state.gen_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = generate_text(prompt, question, model, api_key, temperature, max_tokens)
                    st.markdown(answer)
                    st.session_state.gen_messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
