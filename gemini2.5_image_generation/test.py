from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-pro",
    api_key=api_key,
    temperature=0.7,
)

messages = [
    (
        "system",
        "あなたは非常に有能なアシスタントです。",
    ),
    ("human", "pythonについて教えてください。"),
]
ai_msg = llm.invoke(messages)

print(ai_msg)