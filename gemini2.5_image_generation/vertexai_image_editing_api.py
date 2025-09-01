import os
import base64
from io import BytesIO
import datetime
from dotenv import load_dotenv, find_dotenv
from PIL import Image
from google import genai
from google.genai.types import GenerateContentConfig, Part
import google.auth

_ = load_dotenv(find_dotenv())


SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
credentials, project_id = google.auth.default(scopes=SCOPES)


def convert_to_base64(file_path):
    """
    ファイルをbase64文字列に変換する関数
    file_pathは、変換するファイルのパス
    """
    # ファイルを読み込み
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    base64_str = base64.b64encode(file_bytes).decode("utf-8")

    return base64_str

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

def save_image_from_base64(base64_str, file_path):
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

def process_dict_str_and_image(contents, file_path):
    """
    LLMの出力結果を整理したcontentsを受け取り、画像を保存、テキストを表示する
    contentsは、出力がテキストの場合は`str`キーをもち、画像の場合は`base64`キーを持つ辞書のリスト
    例: [{"str": "出力テキスト"}, {"base64": "base64文字列"}]
    """

    print("============ 生成結果 =============")
    
    # 画像の有無を確認
    has_image = any('base64' in res for res in contents)
    if not has_image:
        print("⚠️  LLMの出力に画像が含まれていません")
    
    for res in contents:
        if 'base64' in res:
            save_image_from_base64(res['base64'], file_path)
        else:
            print("出力テキスト:", res['str'])
    print("===================================")


def validate_and_extract_base64(response):
    """
    LLMの生の出力結果responseを受け取り、base64文字列や生成されたテキストを抽出して辞書のリストを返す関数
    responseは、Google GenAI APIからの応答
    最終的な出力は、LLM出力がテキストの場合は`str`キーをもち、画像の場合は`base64`キーを持つ辞書のリスト
    # 例: [{"str": "出力テキスト"}, {"base64": "base64文字列"}]
    """
    # responseにcandidates属性があるか確認
    if not hasattr(response, 'candidates'):
        raise ValueError("responseにcandidates属性が存在しません。")

    #print(f"response: {response}")
    
    # candidatesが存在し、要素があるか確認
    if not response.candidates or len(response.candidates) == 0:
        raise ValueError("response.candidatesが空です。")

    # 最初のcandidateのcontentを取得
    candidate = response.candidates[0]
    if not hasattr(candidate, 'content'):
        raise ValueError("candidateにcontent属性が存在しません。")
        
    content = candidate.content
    if not hasattr(content, 'parts'):
        raise ValueError("contentにparts属性が存在しません。")

    # contentのpartsがリストであり、要素があるか確認
    if not content.parts or len(content.parts) == 0:
        raise ValueError("content.partsが空です。")

    extracted_list = []

    for idx, part in enumerate(content.parts):
        if hasattr(part, 'text') and part.text:
            # テキスト部分の場合
            extracted_list.append({"str": part.text})
            
        elif hasattr(part, 'inline_data') and part.inline_data:
            # 画像データの場合
            image_data = part.inline_data.data
            
            # データがbase64エンコードされている場合はそのまま使用
            if isinstance(image_data, str):
                base64_str = image_data
            else:
                # バイナリデータの場合はbase64エンコード
                base64_str = base64.b64encode(image_data).decode('utf-8')
            
            extracted_list.append({"base64": base64_str})
        else:
            raise ValueError(f"part[{idx}] は想定外の形式です。")

    return extracted_list


def main():
    # クライアントの定義
    client = genai.Client(vertexai=True, project=project_id, location="global")
    MODEL_ID = "gemini-2.5-flash-image-preview"

    # ==========　一度に生成する生成枚数の指定 ==========
    generate_images = 1

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
    query = "リアルな3dフィギュアにしてください。ただしポーズをもっとかっこいい女の子のポーズにしてください。"
    #query = "asapというアカウントのキャラクターなので、カッコよくasapというロゴを入れてください。おしゃれかつ違和感のないようにお願いします。"

    # =================================

    # 画像をbase64に変換
    file_b64 = convert_to_base64(file_path)
    print("ファイルのbase64変換が完了したので、処理を開始します。")

    for i in range(generate_images):
        print(f"画像{i+1}の生成を開始します。")
        # 画像の編集
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                Part.from_text(text = query),
                Part.from_bytes(data=base64.b64decode(file_b64), mime_type="image/png")
            ],
            config=GenerateContentConfig(
                system_instruction=(
                    "# 目的\n"
                    "あなたのタスクは画像編集です。ユーザが入力した画像を元に、ユーザが指定した内容で新しい画像を生成してください。\n"
                    "\n"
                    "# ルール\n"
                    "ユーザが指示した内容に関係のない物体は、元の画像と全く同一にしてください。\n"
                    "ユーザが指示した内容だけをユーザの指示に忠実に編集して、画像を生成してください。\n"
                    "ユーザからの指示が変更依頼の場合は、そのオブジェクトと指定されたオブジェクトを元の画像から入れ替える形で編集してください。\n"
                    "ユーザからの指示が消去依頼の場合は、そのオブジェクトを元の画像から消去してください。\n"
                    "ユーザからの指示が追加依頼の場合は、そのオブジェクトを元の画像に追加してください。\n"
                ),
                temperature=0.7,
                response_modalities=["TEXT", "IMAGE"],
                candidate_count=1,
            ),
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