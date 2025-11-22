import streamlit as st

# Page Configuration
st.set_page_config(page_title="QnA ChatBot", page_icon="🤖", layout="wide")

# Define Pages
home_page = st.Page("pages/home.py", title="Home", icon="🏠", default=True)
search_page = st.Page("pages/search.py", title="Search Engine", icon="🔍")
db_page = st.Page("pages/chat_db.py", title="Database Chat", icon="🗄️")

# Navigation
pg = st.navigation({
    "Main": [home_page],
    "Tools": [search_page, db_page]
})

# Run Navigation
pg.run()
