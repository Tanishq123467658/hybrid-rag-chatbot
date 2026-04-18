import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from rag_agent import agent, DocumentLoader

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages = []

while True:
    prompt = input("\nEnter prompt (type 'load' to load PDF, 0 to exit): ").strip()

    if prompt == "0":
        break

    if not prompt:
        print("Please enter something.")
        continue

    # LOAD
    if prompt.lower() == "load":
        file_path = input("Enter PDF path: ").strip()
        result = DocumentLoader.invoke({"file_path": file_path})
        print("Bot:", result)
        continue

    messages.append(HumanMessage(content=prompt))

    response = agent(model=model, messages=messages)

    if not response or response.strip() == "":
        response = "I couldn't generate a response."

    print("Bot:", response)

    messages.append(AIMessage(content=response))