import os
import base64
from io import BytesIO
import datetime
from dotenv import load_dotenv, find_dotenv
from PIL import Image
import sys

import google.auth
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_google_vertexai import ChatVertexAI, Modality


_ = load_dotenv(find_dotenv())
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
credentials, project_id = google.auth.default(scopes=SCOPES)

def convert_to_base64(file_path:str):
    """
    ファイルをbase64文字列に変換する関数
    file_pathは、変換するファイルのパス
    """

    # ファイルを読み込み
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    base64_str = base64.b64encode(file_bytes).decode("utf-8")

    return base64_str

def base64_to_image(base64_str:str):
    """
    base64文字列を受け取り、画像に変換する関数
    base64_strは、base64文字列
    """
    # base64文字列をデコード
    img_data = base64.b64decode(base64_str)
    
    # BytesIOオブジェクトを作成
    img = Image.open(BytesIO(img_data))
    
    return img

def save_image_from_base64(base64_str:str, file_path:str):
    """
    base64文字列を受け取り、画像を保存する関数
    base64_strは、base64文字列
    file_pathは、元画像のパスで、保存した画像に元画像の名前を利用する。例： outputs/sample1/sample1_2023-10-01_12-00-00.png
    """ 
    # 画像データにデコード
    img = base64_to_image(base64_str)

    # ファイル名を生成　元画像名＋タイムスタンプ
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"outputs/{os.path.basename(file_path).split('.')[0]}/{os.path.basename(file_path).split('.')[0]}_{timestamp}.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 画像を保存
    img.save(filename)
    print(f"画像を保存しました: {filename}")

def process_dict_str_and_image(contents:list, file_path:str):
    """
    LLMの出力結果を整理したcontentsを受け取り、画像を保存、テキストを表示する
    contentsは、出力がテキストの場合は`str`キーをもち、画像の場合は`base64`キーを持つ辞書のリスト
    例: [{"str": "出力テキスト"}, {"base64": "base64文字列"}]
    """
    
    print("============ 生成結果 =============")
    for res in contents:
        if 'base64' in res:
            save_image_from_base64(res['base64'], file_path)
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

    # response(LLMの出力)にcontent属性があるか確認
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
    model = ChatVertexAI(
        model_name="gemini-2.5-flash-image-preview",
        credentials=credentials,
        project=project_id,
        location="global",
        temperature=0.7,
        response_modalities=[Modality.IMAGE, Modality.TEXT]
        )

    # messageを作成する
    message = [
        SystemMessage(content= """
# 目的
あなたのタスクは画像編集です。ユーザが入力した画像を元に、ユーザが指定した内容で新しい画像を生成してください。

# ルール
ユーザが指示した内容に関係のない物体は、元の画像と全く同一にしてください。
ユーザが指示した内容だけをユーザの指示に忠実に編集して、画像を生成してください。
ユーザからの指示が変更依頼の場合は、そのオブジェクトと指定されたオブジェクトを元の画像から入れ替える形で編集してください。
ユーザからの指示が消去依頼の場合は、そのオブジェクトを元の画像から消去してください。
ユーザからの指示が追加依頼の場合は、そのオブジェクトを元の画像に追加してください。
    """),
        HumanMessagePromptTemplate.from_template(
            [
                {
                    "type": "text",
                    "text": "{user_input}"
                },
                {
                    "image_url": {
                        "url": "data:image/png;base64,{image}"

                    }
                }
            ]
        ),
        #AIMessage(content="承知いたしました。画像編集をいたしました。"),
        #HumanMessage(content="画像が出力されていません。"),
    ]

    # messageからプロンプトを作成
    prompt = ChatPromptTemplate.from_messages(message)

    # chainを作成
    chain = prompt | model


    # ==========　一度に生成する生成枚数の指定 ==========
    generate_images = 10


    # ========== 読み込む画像のパス ==========
    #file_path = "inputs/sample1.png"
    #file_path = "inputs/sample2.png"
    file_path = "inputs/images3.png"


    # ========== 編集内容の指定 ==========
    #query = "女性の表情を溢れるくらいの笑顔に変更して"
    #query = "表情を笑顔に変更して"
    #query = "空を夕焼けの空に変更して、ベンチの色を赤色に変更して"
    #query = "コーヒーカップを消去して"
    #query = "画像に写っている女の子を元にした漫画を作成してください。4コマ漫画で、楽しそうに友達と遊んでいるところがいいです。"
    query = "3dフィギュアにしてください。ただしポーズをもっとかっこいい女の子のポーズにしてください。"
    #query = "asapというアカウントのキャラクターなので、カッコよくasapというロゴを入れてください。おしゃれかつ違和感のないようにお願いします。"


    # =================================

    # 画像をbase64に変換
    file_b64 = convert_to_base64(file_path)
    print("ファイルのbase64変換が完了したので、処理を開始します。")

    for i in range(generate_images):
        print(f"画像{i+1}の生成を開始します。")
        # 画像の生成
        response = chain.invoke(
            {"user_input": query, "image": file_b64}, 
            generation_config=dict(response_modalities=["TEXT", "IMAGE"])
            )


        # LLMの出力結果が正しく画像になっているかバリデーション
        # その後、画像を保存
        try:
            image_str_dict = validate_and_extract_base64(response)
            # 画像を保存、テキストは表示
            process_dict_str_and_image(image_str_dict, file_path)

        except ValueError as e:
            print(f"エラー発生: {e}")


    

if __name__ == "__main__":
    main()
