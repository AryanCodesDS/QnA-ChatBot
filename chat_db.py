import sqlite3
from langchain_community.agent_toolkits.sql.base import create_sql_agent

from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq

import streamlit as st
from create_db import DBconnection
from sqlite3 import Error
import tempfile

#Streamlit App Configuration
st.set_page_config(page_title="Chat with your database", page_icon=":guardsman:", layout="wide")
st.title("Chat with your database")

#Sidebar
radio_opt = ["Upload a sqlite3 DB","Create a test sqlite3 DB","Connect to a localDB"]
selected_opt = st.sidebar.radio(
    label = "Select a Database",
    options= radio_opt
)

#API Key
api_key = st.sidebar.text_input(label="Enter Groq API Key") or "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"

list_models = [
               "llama-3.3-70b-versatile",
               "llama-3.1-8b-instant",
               "llama3-70b-8192",
               "llama3-8b-8192",
               "mixtral-8x7b-32768"
               ]
model = st.selectbox("Select A Model",options = list_models)
llm = ChatGroq(model=model,api_key=api_key,streaming=True)

if 'db_connection' not in st.session_state:
    st.session_state.db_connection = None

#Performing operations based on selected option
if selected_opt == radio_opt[0]:
    sqlitedb = st.sidebar.file_uploader(label="Upload a sqlite3 Database",type=["db","sqlite3"])
    
    if sqlitedb is not None:
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            # Write the uploaded content to the temp file
                tmp_file.write(sqlitedb.getvalue())
                tmp_file_path = tmp_file.name
        
            # Connect to the database
            obj = DBconnection("sqlite", database=tmp_file_path)
            st.session_state.db_connection = obj.get_engine()
            st.success("Connected to SQLite database successfully!")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
if selected_opt == radio_opt[2]:

    host = st.sidebar.text_input(label="Enter Host Name",)
    user = st.sidebar.text_input(label="Enter User Name")
    password = st.sidebar.text_input(label="Enter Password",type="password")
    database = st.sidebar.text_input(label="Enter Databse Name")
    list_opt = [host,user,password,database]
    
    
    flag = True
    if '' not in list_opt :
        flag = False

    rdbms = st.sidebar.selectbox(label="Select RDBMS",options=["MySQL","PostgreSQL"],disabled=flag)
    try:
        if st.sidebar.button("Connect",):
            if rdbms == "MySQL":
                st.session_state.db_connection = DBconnection("mysql", host, user, password, database).get_engine()
                st.success("Connected to MySQL database successfully!")
                    
            elif rdbms == "PostgreSQL":
                st.session_state.db_connection = DBconnection("psql", host, user, password, database).get_engine()
                st.success("Connected to PostgreSQL database successfully!")

        if st.sidebar.button("Close connection"):
            st.session_state.db_connection = None          
    except:
        st.toast("Wrong Credentials,check again and enter.")

#Toolkit

if st.session_state.db_connection is not None:
    tk = SQLDatabaseToolkit(db = st.session_state.db_connection,llm=llm)

    agent = create_sql_agent(
        llm=llm,
        toolkit=tk,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=10,

    )

    if "messages" not in st.session_state or st.sidebar.button("Clear chat history"):
        st.session_state["messages"] = [{
            "role" : "assistant",
            "content": "How can i help you?"
        }]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])


    user_query = st.chat_input("Ask anything about your database")

    if user_query:
        st.session_state["messages"].append({
            "role" : "user",
            "content": user_query
        })
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            resp = agent.run(user_query,callbacks=[st_callback])
            st.session_state["messages"].append({
                "role" : "assistant",
                "content": resp
            })
            st.write(resp)

else:
    st.warning("Please connect to a database before using the chatbot.")

########### EOS