import os
import base64
from io import BytesIO
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_APIKEY")

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
    pdf_file_path = data["pdf_file_path"]

    message = [
        SystemMessage(content="あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。また考える過程も出力してください。"),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"{user_input}"
                },
                {
                    "type": "file",
                    "file": {
                        "filename": f"{pdf_file_path}",
                        "file_data": f"data:application/pdf;base64,{pdf}"
                    }
                },
            ]
        )
    ]

    return message  

model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    temperature=0.001,
    top_p=0.001
)

"""model = ChatOpenAI(
    model="o1",
    openai_api_key=api_key,
)"""

chain = prompt_func | model | StrOutputParser()

file_path = "inputs/DeepSeek-R1-paper-asap-r3.pdf"
pdf_b64 = convert_pdf_to_base64(file_path)

print("pdfファイルの変換が完了したので、処理を開始します。")

#query = "PDFは何を解説しているか教えてください。"
query = "5ページ目の年表を説明してください。図と説明をつなぐ線に着目して、時系列がずれないように正確に解説してください。"

#以下でも良い
#output = chain.invoke({"user_input": query, "pdf": pdf_b64, "pdf_file_path": file_path})

#stream出力
output = ""
for chunk in chain.stream({"user_input": query, "pdf": pdf_b64, "pdf_file_path": file_path}):
    print(chunk, end="", flush=True)
    output += chunk

print("\n=== Output ===")
print(output)

