import os
import base64
from io import BytesIO
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv, find_dotenv

def convert_pdf_to_base64(pdf_path):
    """
    Convert a PDF file to a Base64 encoded string.

    :param pdf_path: path to the pdf file
    :return: Base64 string
    """
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    pdf_str = base64.b64encode(pdf_bytes).decode("utf-8")
    return pdf_str

def prompt_func(data):
    user_input = data["user_input"]
    pdf = data["pdf"]

    message = [
        SystemMessage(content="あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。また考える過程も出力してください。"),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"{user_input}"
                },
                {
                    'type': 'media',
                    'mime_type': "application/pdf",
                    'data': pdf
                },
            ]
        )
    ]

    return message  

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.001,
    top_p=0.001
)

chain = RunnableLambda(prompt_func) | model | StrOutputParser()

file_path = "inputs/DeepSeek-R1-paper-asap-r3.pdf"
pdf_b64 = convert_pdf_to_base64(file_path)

print("pdfファイルの変換が完了したので、処理を開始します。")

query = "PDFは何を解説しているか教えてください。"

output = ""
for chunk in chain.stream({"user_input": query, "pdf": pdf_b64}):
    print(chunk, end="", flush=True)
    output += chunk

print("\n=== Output ===")
print(output)
