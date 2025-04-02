import os
import base64
from io import BytesIO
import datetime
from dotenv import load_dotenv, find_dotenv
from PIL import Image
import sys

from langchain_google_genai import ChatGoogleGenerativeAI, Modality
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage

_ = load_dotenv(find_dotenv())
api_key = os.getenv("GOOGLE_API_KEY")


def base64_to_image(base64_str):
    """
    base64文字列を受け取り、画像に変換する関数
    base64_strは、base64文字列
    """
    # base64文字列をデコード
    img_data = base64.b64decode(base64_str)
    
    # BytesIOオブジェクトを作成
    img = Image.open(BytesIO(img_data))
    
    return img

def save_image_from_base64(base64_str):
    """
    base64文字列を受け取り、画像を保存する関数
    base64_strは、base64文字列
    """ 
    # 画像データにデコード
    img = base64_to_image(base64_str)

    # ファイル名を生成　元画像名＋タイムスタンプ
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"outputs/new_generation/generate_{timestamp}.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 画像を保存
    img.save(filename)
    print(f"画像を保存しました: {filename}")

def process_dict_str_and_image(contents):
    """
    LLMの出力結果を整理したcontentsを受け取り、画像を保存、テキストを表示する
    contentsは、出力がテキストの場合は`str`キーをもち、画像の場合は`base64`キーを持つ辞書のリスト
    例: [{"str": "出力テキスト"}, {"base64": "base64文字列"}]
    """

    print("============ 生成結果 =============")
    for res in contents:
        if 'base64' in res:
            save_image_from_base64(res['base64'])
        else:
            print("出力テキスト:", res['str'])
    print("===================================")


def validate_and_extract_base64(response):
    """
    LLMの生の出力結果responseを受け取り、base64文字列や生成されたテキストを抽出して辞書のリストを返す関数
    responseは、LLMからの応答
    最終的な出力は、LLM出力がテキストの場合は`str`キーをもち、画像の場合は`base64`キーを持つ辞書のリスト
    # 例: [{"str": "出力テキスト"}, {"base64": "base64文字列"}]
    """
    # responseにcontent属性があるか確認
    if not hasattr(response, 'content'):
        raise ValueError("responseにcontent属性が存在しません。")

    content = response.content

    # contentがリストであり、要素があるか確認
    if not isinstance(content, list) or len(content) == 0:
        print(f'生成結果:"{content}"')
        raise ValueError("response.contentが空、またはリスト型ではありません。")

    extracted_list = []

    for idx, item in enumerate(content):
        if isinstance(item, str):
            # str型の場合、そのまま辞書に格納
            extracted_list.append({"str": item})

        elif isinstance(item, dict):
            image_url = item.get('image_url')

            # 辞書内に'image_url'キーがあるか確認
            if not isinstance(image_url, dict):
                raise ValueError(f"content[{idx}] の image_urlが辞書型ではありません。（型: {type(image_url)}）")

            url = image_url.get('url')

            if not isinstance(url, str):
                raise ValueError(f"content[{idx}] の urlが文字列型ではありません。（型: {type(url)}）")

            if ',' not in url:
                raise ValueError(f"content[{idx}] の urlにカンマが含まれていません。")

            # base64部分を抽出して辞書に格納
            base64_str = url.split(',')[-1]
            extracted_list.append({"base64": base64_str})

        else:
            raise ValueError(f"content[{idx}] は想定外の型です。（型: {type(item)}）")

    return extracted_list


def main():
    # モデルの定義。APIキーは環境変数から取得
    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash-exp-image-generation",
        google_api_key=api_key,
        response_modalities=[Modality.IMAGE, Modality.TEXT]
        )

    # messageを作成する
    message = [
        # 現在、gemini-2.0-flash-exp-image-generationではsystem_messageを利用できないので、HumanMessageを利用する
        HumanMessage(content= """
# 目的
あなたのタスクは画像生成です。ユーザが指定した内容で新しい画像を生成してください。

----以下がユーザの入力です----
    """),
        HumanMessagePromptTemplate.from_template(
            [
                {
                    "type": "text",
                    "text": "{user_input}"
                },
            ]
        ),
    ]

    # messageからプロンプトを作成
    prompt = ChatPromptTemplate.from_messages(message)

    # chainを作成
    chain = prompt | model


    # ==========　一度に生成する生成枚数の指定 ==========
    # RPMは10なのでそれを考慮して設定してください
    generate_images = 5



    # ========== 生成内容の指定 ==========
    query = "家のPCデスクの実写画像を作成してください。机の上にはノートPCとコーヒーカップを置いてください。机の色は黒色でお願いします。そのほかは一般的な部屋の様子でいい感じに作ってください。"


    # =================================

    for i in range(generate_images):
        print(f"画像{i+1}の生成を開始します。")
        # 画像の生成
        response = chain.invoke(
            {"user_input": query}, 
            generation_config=dict(response_modalities=["TEXT", "IMAGE"])
            )


        # LLMの出力結果が正しく画像になっているかバリデーション
        # その後、画像を保存
        try:
            image_str_dict = validate_and_extract_base64(response)
            # 画像を保存、テキストは表示
            process_dict_str_and_image(image_str_dict)

        except ValueError as e:
            print(f"エラー発生: {e}")


    

if __name__ == "__main__":
    main()
