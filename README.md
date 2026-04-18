# 📄 Hybrid RAG Chatbot

A hybrid AI chatbot that combines **document-based retrieval (RAG)** with **real-time web search** to answer user queries more accurately.

Built using **LangChain, Streamlit, ChromaDB, and Tavily**, this project allows users to upload PDFs and ask questions, while also falling back to web search when the answer isn’t found in the documents.

---

## 🚀 Features

* 📂 Upload and chat with **multiple PDFs**
* 🔍 **RAG (Retrieval-Augmented Generation)** using ChromaDB
* 🌐 **Web search fallback** using Tavily API
* 🤖 Agent-based architecture with tool usage
* 💬 Chat-style UI built with Streamlit
* 📌 Source-aware responses (`PDF` or `WEB`)
* 🧠 Context-aware multi-turn conversation

---

## 🧠 How It Works

1. User uploads a PDF

2. Document is:

   * Loaded
   * Split into chunks
   * Converted into embeddings
   * Stored in ChromaDB

3. When a question is asked:

   * Agent first queries **AskDocs (RAG)**
   * If no relevant info → falls back to **Tavily web search**
   * Final answer is generated using the retrieved context

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Google Gemini (via LangChain)
* **Embeddings:** Mistral AI
* **Vector DB:** ChromaDB
* **Search API:** Tavily
* **Framework:** LangChain Agents

---

## 📂 Project Structure

```
.
├── app.py              # Streamlit UI
├── rag_agent.py        # Agent + tools (RAG + Web)
├── chroma_db/          # Vector databases (auto-generated)
├── .env                # API keys
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hybrid-rag-chatbot.git
cd hybrid-rag-chatbot
```

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Add environment variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

### 5. Run the app

```bash
streamlit run app.py
```

---

## 💡 Example Use Cases

* 📚 Ask questions from lecture notes or PDFs
* 📄 Analyze reports and documents
* 🌐 Get real-time answers when data isn’t in the file
* 🧠 Build intelligent assistants with hybrid retrieval

---

## ⚠️ Limitations

* Agent behavior is not fully deterministic
* Retrieval quality depends on chunking and embeddings
* Large PDFs may increase processing time

---

## 🔮 Future Improvements

* 📌 Show exact document sources (citations)
* 📊 Rank results by relevance score
* 📁 Document management (delete/view files)
* ⚡ Streaming responses
* 🌍 Deploy as a web app

---

## 👨‍💻 Author

**Tanishq Battul**

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub — it helps a lot!
