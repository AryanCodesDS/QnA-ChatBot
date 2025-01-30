import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS

from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_core.runnables import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import concurrent.futures
# Load langchain libs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

from langchain.embeddings.base import Embeddings
from typing import List

load_dotenv()
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed = SentenceTransformerEmbeddings(model_name)

list_models = ["gemma2-9b-it",
               "llama-3.3-70b-versatile",
               "llama-3.1-8b-instant",
               "llama3-70b-8192",
               "llama3-8b-8192",
               "mixtral-8x7b-32768"
               ]


def generate_text(prompt, question, model, api_key, temperature, max_tokens):
    llm = ChatGroq(model=model,
                   api_key=api_key,
                   temperature=temperature,
                   max_tokens=max_tokens)
    chain = prompt | llm | StrOutputParser()
    res = chain.invoke({"question": question})
    return res


def process_pdf(pdf):
    temp_pdf = f'./temp_{pdf.name}.pdf'
    with open(temp_pdf, 'wb') as f:
        f.write(pdf.getvalue())
    pdf_loader = PyPDFLoader(temp_pdf)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    docs = pdf_loader.load_and_split(text_splitter=text_splitter)
    os.remove(temp_pdf)
    return docs


# Title of Application
st.title("Question Answering Using LLMs")


@st.cache_data
def load_and_process_pdfs(pdf_files):
    documents = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_pdf, pdf_files)
        for r in results:
            documents.extend(r)
    return documents

# Sidebar
st.sidebar.title("Customise your model")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
model = st.sidebar.selectbox("Select A Model", list_models)
temperature = st.sidebar.slider("Select Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.slider("Select Max Tokens", 0, 1000, 100)
chat_with_pdf_option = st.sidebar.checkbox("Chat with PDF")
if chat_with_pdf_option:
    st.sidebar.write("Upload a PDF file")
    pdf_file = st.sidebar.file_uploader(
        "Upload PDF", type=["pdf"], accept_multiple_files=True)
    session_id = st.sidebar.text_input(
        "Enter Session ID", value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store = {}
    # ... (in the main code)
    if pdf_file:
        documents = load_and_process_pdfs(pdf_file)
        st.write("PDF Uploaded Successfully")
        vector_store = FAISS.from_documents(documents, embedding=embed)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})


def RAG_conv(prompt, question, model, api_key, temperature, max_tokens):
    if pdf_file:
        pdfprompt = (
            """
            Answer the questions based on provided context only.
            please provide accurate response based on question and context.
            {context}
            """
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", pdfprompt),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ]
        )

        llm = ChatGroq(model=model,
                       api_key=api_key,
                       temperature=temperature,
                       max_tokens=max_tokens)
        history_aware_retrieval_chain = create_history_aware_retriever(
            llm, retriever, prompt)
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(
            history_aware_retrieval_chain, qa_chain)
        conrag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history=getsession, history_messages_key="chat_history", input_messages_key="input", output_messages_key="answer")

        res = conrag_chain.invoke({"input": question}, config={
            "configurable": {"session_id": session_id}
        })

        return res["answer"]
    else:
        return "Please upload a PDF file"


def getsession(session_id) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


# Main Page
if chat_with_pdf_option:
    st.write("Chat with PDF using LLMs")
    template = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history,"
        "formulate a standalone question which can be understood"
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
    question = st.text_input("You:")
    if question:
        session_history = getsession(session_id)

        answer = RAG_conv(prompt, question, model,
                          api_key, temperature, max_tokens)
        st.write(st.session_state.store)
        st.write("AI:", answer)
        st.write("Chat History:", session_history.messages)
    else:
        st.write("AI: Please ask a question")
else:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are helpful question answering AI"),
            ("user", "{question}"),
        ]
    )

    st.write("This is a simple question answering application using LLMs. You can ask any question and the AI will try to answer it.")
    question = st.text_input("You:")
    if question:
        answer = generate_text(prompt, question, model,
                               api_key, temperature, max_tokens)

        st.write("AI:", answer)

    else:
        st.write("AI: Please ask a question")
