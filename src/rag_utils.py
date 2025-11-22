from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import streamlit as st

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if "store" not in st.session_state:
        st.session_state.store = {}
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def generate_text(prompt, question, model, api_key, temperature, max_tokens):
    llm = ChatGroq(model=model,
                   api_key=api_key,
                   temperature=temperature,
                   max_tokens=max_tokens)
    chain = prompt | llm | StrOutputParser()
    res = chain.invoke({"question": question})
    return res

def rag_conversation(prompt, question, model, api_key, temperature, max_tokens, retriever, session_id):
    pdfprompt = (
        """
        Answer the questions based on provided context only.
        Please provide accurate response based on question and context.
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
        rag_chain, 
        get_session_history=get_session_history, 
        history_messages_key="chat_history", 
        input_messages_key="input", 
        output_messages_key="answer"
    )

    res = conrag_chain.invoke({"input": question}, config={
        "configurable": {"session_id": session_id}
    })

    return res["answer"]
