import os
import base64
from io import BytesIO
import datetime
from dotenv import load_dotenv, find_dotenv
from PIL import Image
from google import genai
from google.genai.types import GenerateContentConfig
import google.auth

_ = load_dotenv(find_dotenv())


SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
credentials, project_id = google.auth.default(scopes=SCOPES)


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
    
    # 画像の有無を確認
    has_image = any('base64' in res for res in contents)
    if not has_image:
        print("⚠️  LLMの出力に画像が含まれていません")
    
    for res in contents:
        if 'base64' in res:
            save_image_from_base64(res['base64'])
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

    print(f"response: {response}")
    
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
    # RPMは10なのでそれを考慮して設定してください
    generate_images = 1

    # ========== 生成内容の指定 ==========
    query = "家のPCデスクの実写画像を作成してください。机の上にはノートPCとコーヒーカップを置いてください。机の色は黒色でお願いします。そのほかは一般的な部屋の様子でいい感じに作ってください。"
    #query = "家のPCデスクの実写画像"

    # =================================

    for i in range(generate_images):
        print(f"画像{i+1}の生成を開始します。")
        # 画像の生成
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=(
                query
            ),
            config=GenerateContentConfig(
                system_instruction=(
                    "# 目的\n",
                    "あなたのタスクは画像生成です。ユーザが指定した内容で新しい画像を生成してください。"
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
            process_dict_str_and_image(image_str_dict)

        except ValueError as e:
            print(f"エラー発生: {e}")


if __name__ == "__main__":
    main()