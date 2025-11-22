import streamlit as st
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.tools import TavilySearchResults

st.title("🔍 Search Engine Using LLMs")

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("Enter Groq API Key", type="password")
    tavily_api_key = st.text_input("Enter Tavily API Key", type="password")
    
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

# Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

# Tavily Search Tool
search = None
if tavily_api_key:
    search = TavilySearchResults(tavily_api_key=tavily_api_key)

tools = [arxiv_tool, wikipedia_tool]
if search:
    tools.append(search)

# Chat Interface
if "search_messages" not in st.session_state:
    st.session_state.search_messages = [
        {"role": "assistant", "content": "Hello, I am a chatbot who can search the web. How can I help you today?"}
    ]

for msg in st.session_state.search_messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input(placeholder="Type here..."):
    st.session_state.search_messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
    elif not tavily_api_key:
        st.error("Please enter your Tavily API Key in the sidebar for web search.")
    else:
        try:
            llm = ChatGroq(model=model, api_key=api_key, streaming=True)
            
            # Create ReAct Agent
            template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

            prompt = PromptTemplate.from_template(template)
            
            agent = create_react_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = agent_executor.invoke({"input": user_input}, config={"callbacks": [st_cb]})
                st.session_state.search_messages.append({"role": "assistant", "content": response["output"]})
                st.write(response["output"])
        except Exception as e:
            st.error(f"An error occurred: {e}")
