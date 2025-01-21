import streamlit as st
from dotenv import load_dotenv
import os

#Load langchain libs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq

load_dotenv()

list_models = ["gemma2-9b-it",
"llama-3.3-70b-versatile",
"llama-3.1-8b-instant",
"llama3-70b-8192",
"llama3-8b-8192",
"mixtral-8x7b-32768"
]

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","you are helpful question answering AI"),
        ("user","{question}"),
    ]
)
def generate_text(prompt,question,model,api_key,temperature,max_tokens):
    llm = ChatGroq(model=model,
                   api_key=api_key,
                   temperature=temperature,
                   max_tokens=max_tokens)
    chain = prompt|llm|StrOutputParser()
    res = chain.invoke({"question":question})
    return res

#Title of Application
st.title("Question Answering Using LLMs")

#Sidebar
st.sidebar.title("Customise your model")
api_key = st.sidebar.text_input("Enter Groq API Key",type="password")
model = st.sidebar.selectbox("Select A Model", list_models)
temperature = st.sidebar.slider("Select Temperature",0.0,1.0,0.5)
max_tokens = st.sidebar.slider("Select Max Tokens",0,1000,100)

#Main Page
st.write("This is a simple question answering application using LLMs. You can ask any question and the AI will try to answer it.")
question=st.text_input("You:")

if question:
    answer = generate_text(prompt,question,model,api_key,temperature,max_tokens)
    st.write("AI:",answer)
else:
    st.write("AI: Please ask a question")