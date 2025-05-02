import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

#importing Arxiv and wikipedia s tools and their API wrappers.
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_google_community.search import GoogleSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper

#Langchain Agents
from langchain.agents import initialize_agent,AgentType
from langchain_groq import ChatGroq
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

#Arxiv Wrapper
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=1000)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

#Wikipedia Wrapper
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=1000)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

search=DuckDuckGoSearchRun()

#Main
st.title("Search Engine Using LLMs")
#Sidebar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter Groq API Key")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, I am a chatbot who can search the web.How can I help you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input(placeholder="Type here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    llm = ChatGroq(model="llama3-8b-8192", api_key=api_key,streaming=True)
    tools = [arxiv_tool,wikipedia_tool,search]

    search_agent = initialize_agent(agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, llm=llm, tools=tools,handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

