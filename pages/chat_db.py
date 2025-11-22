import streamlit as st
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq
import tempfile
import os

# Import from src
from src.database import DBconnection

# Streamlit App Configuration
# st.set_page_config is handled in app.py
st.title("🗄️ Chat with your database")

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    radio_opt = ["Upload a sqlite3 DB", "Connect to a localDB"]
    selected_opt = st.radio(
        label="Select a Database",
        options=radio_opt
    )

    # API Key
    api_key = st.text_input(label="Enter Groq API Key", type="password")

    list_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    model = st.selectbox("Select A Model", options=list_models)

if 'db_connection' not in st.session_state:
    st.session_state.db_connection = None

# Performing operations based on selected option
if selected_opt == radio_opt[0]:
    sqlitedb = st.sidebar.file_uploader(label="Upload a sqlite3 Database", type=["db", "sqlite3"])

    if sqlitedb is not None:
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
                # Write the uploaded content to the temp file
                tmp_file.write(sqlitedb.getvalue())
                tmp_file_path = tmp_file.name

            # Connect to the database
            obj = DBconnection("sqlite", database=tmp_file_path)
            st.session_state.db_connection = obj.get_engine()
            st.sidebar.success("Connected to SQLite database successfully!")

        except Exception as e:
            st.sidebar.error(f"An error occurred: {e}")

elif selected_opt == radio_opt[1]:
    st.sidebar.subheader("Database Credentials")
    host = st.sidebar.text_input(label="Enter Host Name")
    user = st.sidebar.text_input(label="Enter User Name")
    password = st.sidebar.text_input(label="Enter Password", type="password")
    database = st.sidebar.text_input(label="Enter Database Name")
    
    rdbms = st.sidebar.selectbox(label="Select RDBMS", options=["MySQL", "PostgreSQL"])
    
    if st.sidebar.button("Connect"):
        try:
            if rdbms == "MySQL":
                st.session_state.db_connection = DBconnection("mysql", host, user, password, database).get_engine()
                st.sidebar.success("Connected to MySQL database successfully!")

            elif rdbms == "PostgreSQL":
                st.session_state.db_connection = DBconnection("psql", host, user, password, database).get_engine()
                st.sidebar.success("Connected to PostgreSQL database successfully!")
        except Exception as e:
            st.sidebar.error(f"Connection failed: {e}")

    if st.sidebar.button("Close connection"):
        st.session_state.db_connection = None
        st.sidebar.info("Connection closed.")

# Toolkit
if st.session_state.db_connection is not None and api_key:
    llm = ChatGroq(model=model, api_key=api_key, streaming=True)
    tk = SQLDatabaseToolkit(db=st.session_state.db_connection, llm=llm)

    agent = create_sql_agent(
        llm=llm,
        toolkit=tk,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=10,
        handle_parsing_errors=True
    )

    if "db_messages" not in st.session_state or st.sidebar.button("Clear chat history"):
        st.session_state["db_messages"] = [{
            "role": "assistant",
            "content": "How can I help you with your database?"
        }]

    for msg in st.session_state.db_messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input("Ask anything about your database")

    if user_query:
        st.session_state["db_messages"].append({
            "role": "user",
            "content": user_query
        })
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            try:
                resp = agent.run(user_query, callbacks=[st_callback])
                st.session_state["db_messages"].append({
                    "role": "assistant",
                    "content": resp
                })
                st.write(resp)
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    if not api_key:
        st.warning("Please enter your Groq API Key to proceed.")
    elif st.session_state.db_connection is None:
        st.info("Please connect to a database to start chatting.")