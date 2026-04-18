import os
import hashlib
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tavily import TavilyClient
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

# =========================
# EMBEDDINGS
# =========================
embeddings = MistralAIEmbeddings(model="codestral-embed-2505")

# =========================
# TAVILY
# =========================
try:
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
except Exception:
    tavily_client = None

DB_ROOT = "chroma_db"
os.makedirs(DB_ROOT, exist_ok=True)

loaded_collections = set()

# =========================
# DOCUMENT LOADER
# =========================
@tool
def DocumentLoader(file_path: str):
    """
    Load a PDF file, split into chunks, and store embeddings in a unique vector database.
    Each uploaded file is stored separately to support multiple documents.
    """

    file_hash = hashlib.md5(open(file_path, "rb").read()).hexdigest()
    db_path = os.path.join(DB_ROOT, file_hash)

    if db_path in loaded_collections:
        return f"{file_path} already loaded."

    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    ).persist()

    loaded_collections.add(db_path)

    return f"{file_path} loaded successfully."


# =========================
# RAG TOOL
# =========================
@tool
def AskDocs(query: str):
    """
    Search across all uploaded documents.
    ALWAYS call this tool first.

    If no useful info found, return 'NO_RELEVANT_INFO'.
    """

    all_results = []

    for folder in os.listdir(DB_ROOT):
        db_path = os.path.join(DB_ROOT, folder)

        try:
            vs = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )

            retriever = vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 2}
            )

            results = retriever.invoke(query)
            all_results.extend(results)

        except:
            continue

    if not all_results:
        return "NO_RELEVANT_INFO"

    context = "\n\n".join([doc.page_content for doc in all_results])

    if len(context.strip()) < 100:
        return "NO_RELEVANT_INFO"

    return f"[SOURCE: PDF]\n{context}"


# =========================
# WEB TOOL
# =========================
@tool
def getContent(query: str):
    """
    Perform web search using Tavily.
    Use ONLY if AskDocs returns 'NO_RELEVANT_INFO'.
    """

    if tavily_client is None:
        return "[SOURCE: WEB]\nWeb search unavailable."

    result = tavily_client.search(query=query)

    return f"[SOURCE: WEB]\n{str(result)}"


# =========================
# AGENT
# =========================
def agent(model, messages):

    system_prompt = """
    You are a strict tool-using AI assistant.

    RULES:
    1. ALWAYS call AskDocs first.
    2. If AskDocs returns 'NO_RELEVANT_INFO', then call getContent.
    3. NEVER answer from your own knowledge.
    4. ALWAYS mention the source (PDF or WEB).
    """

    agent_executor = create_agent(
        model=model,
        tools=[DocumentLoader, AskDocs, getContent],
        system_prompt=system_prompt
    )

    response = agent_executor.invoke({
        "messages": messages
    })

    last = response["messages"][-1].content

    if isinstance(last, list):
        return " ".join([item.get("text", "") for item in last])

    return last