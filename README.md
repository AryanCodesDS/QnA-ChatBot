# 🤖 QnA ChatBot

A powerful and modular AI-powered application built with **Streamlit**, **LangChain**, and **Groq**. This application offers a suite of tools for Question Answering, Document Interaction, Web Search, and Database Querying.

---

## ✨ Features

### 1. 🏠 General Q&A
- **AI-Powered Chat**: Interact with advanced LLMs (Llama 3, Mixtral, Gemma 2) for general queries.
- **Contextual Awareness**: Maintains chat history for natural conversations.

### 2. 📄 Chat with PDF (RAG)
- **Document Upload**: Upload PDF documents directly to the app.
- **RAG Architecture**: Uses **Retrieval-Augmented Generation** to answer questions based *only* on the uploaded document content.
- **Vector Search**: Powered by **FAISS** and **Sentence Transformers** for efficient similarity search.

### 3. 🔍 AI Search Engine
- **Multi-Source Search**: Integrates **Tavily**, **Arxiv**, and **Wikipedia** for comprehensive web research.
- **ReAct Agent**: Uses a reasoning agent to break down complex queries and fetch accurate information.
- **Source Citations**: Provides links and references for the information retrieved.

### 4. 🗄️ Chat with Database
- **SQL Agent**: Chat with your SQL databases using natural language.
- **Multi-DB Support**: Supports **SQLite**, **MySQL**, and **PostgreSQL**.
- **Schema Awareness**: The agent understands your database schema to construct correct SQL queries.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM Orchestration**: [LangChain](https://www.langchain.com/)
- **LLM Provider**: [Groq API](https://groq.com/) (Llama 3, Mixtral, Gemma)
- **Search API**: [Tavily AI](https://tavily.com/)
- **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings**: [Sentence Transformers](https://www.sbert.net/) (HuggingFace)
- **Database**: SQLite, MySQL, PostgreSQL

---

## � Getting Started

### Prerequisites
- Python 3.10+
- [Groq API Key](https://console.groq.com/) (Free)
- [Tavily API Key](https://tavily.com/) (Free)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/AryanCodesDS/QnA-ChatBot.git
   cd QnA-ChatBot
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

---

## � Usage

1. **Home Page**:
   - Select a model from the sidebar.
   - Enter your **Groq API Key**.
   - Chat generally or toggle **"Chat with PDF"** to upload a document.

2. **Search Engine**:
   - Navigate to **Search Engine** via the sidebar.
   - Enter your **Tavily API Key**.
   - Ask complex questions requiring web access (e.g., "Latest AI news").

3. **Database Chat**:
   - Navigate to **Database Chat**.
   - Upload a SQLite file or connect to a local MySQL/PostgreSQL database.
   - Ask questions like "How many users signed up last week?".

---

## � Project Structure

```
QnA-ChatBot/
├── app.py                  # Main entry point & Navigation
├── requirements.txt        # Project dependencies
├── src/                    # Core logic modules
│   ├── database.py         # Database connection utilities
│   ├── pdf_utils.py        # PDF processing & splitting
│   └── rag_utils.py        # RAG chain & LLM handling
├── pages/                  # Application pages
│   ├── home.py             # Home page (General Chat + PDF)
│   ├── search.py           # Search Engine (Tavily/Arxiv/Wiki)
│   └── chat_db.py          # Database Chat (SQL Agent)
└── README.md               # Project documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**AryanCodesDS** - [GitHub Profile](https://github.com/AryanCodesDS)
