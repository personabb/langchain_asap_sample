import os
import base64
from io import BytesIO
import datetime
from dotenv import load_dotenv, find_dotenv
from PIL import Image
import sys
import requests
import json

_ = load_dotenv(find_dotenv())

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ENDPOINT = "https://openrouter.ai/api/v1"

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
    OpenRouterのAPIレスポンスを受け取り、base64文字列や生成されたテキストを抽出して辞書のリストを返す関数
    responseは、APIからのJSONレスポンス
    最終的な出力は、LLM出力がテキストの場合は`str`キーをもち、画像の場合は`base64`キーを持つ辞書のリスト
    # 例: [{"str": "出力テキスト"}, {"base64": "base64文字列"}]
    
    期待されるレスポンス形式:
    {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "テキストレスポンス",
                "images": [{
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,..."
                    }
                }]
            }
        }]
    }
    """
    
    extracted_list = []
    
    # OpenAI形式のレスポンス構造を確認
    if 'choices' not in response:
        print(f"エラー: 期待されるレスポンス形式ではありません")
        print(f"レスポンス: {json.dumps(response, indent=2)}")
        return extracted_list
    
    for choice in response['choices']:
        message = choice.get('message', {})
        
        # contentフィールド（テキスト）がある場合
        content = message.get('content')
        if content and isinstance(content, str) and content.strip():
            extracted_list.append({"str": content})
        
        # imagesフィールド（画像配列）がある場合
        images = message.get('images')
        if images and isinstance(images, list):
            for image_item in images:
                if isinstance(image_item, dict):
                    # image_url形式の処理
                    if image_item.get('type') == 'image_url':
                        image_url = image_item.get('image_url', {})
                        url = image_url.get('url', '')
                        if url and ',' in url:
                            # data:image/png;base64, の後の部分を抽出
                            base64_str = url.split(',')[-1]
                            extracted_list.append({"base64": base64_str})
                            print(f"画像データを検出しました（{len(base64_str)}文字）")
                    # 他の画像形式にも対応（将来の拡張用）
                    elif 'base64' in image_item:
                        extracted_list.append({"base64": image_item['base64']})
                        print(f"画像データを検出しました（直接base64形式）")
    
    if not extracted_list:
        print("警告: レスポンスからテキストも画像も抽出できませんでした")
    
    return extracted_list

def call_openrouter_api(model_name, system_prompt, user_query, image_base64, api_key, endpoint, temperature=0.7):
    """
    OpenRouter APIを直接呼び出す関数
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": temperature,
        # 画像生成を有効にする - OpenRouterの形式に従う
        "modalities": ["image", "text"]
    }
    
    try:
        response = requests.post(
            f"{endpoint}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60  # タイムアウト設定
        )
        
        # ステータスコードの確認
        if response.status_code != 200:
            print(f"APIエラー: ステータスコード {response.status_code}")
            print(f"レスポンス: {response.text}")
            return None
        
        # JSONレスポンスを返す
        result = response.json()
        
        # デバッグ情報の出力
        print(f"APIステータス: 成功")
        if 'usage' in result:
            usage = result['usage']
            print(f"トークン使用量: 入力={usage.get('prompt_tokens', 0)}, 出力={usage.get('completion_tokens', 0)}")
            # 画像トークンの確認
            if 'completion_tokens_details' in usage:
                details = usage['completion_tokens_details']
                if 'image_tokens' in details:
                    print(f"画像トークン: {details['image_tokens']}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"API呼び出しエラー: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSONパースエラー: {e}")
        print(f"レスポンステキスト: {response.text}")
        return None

def main():
    print(f"API Key: {'設定済み' if API_KEY else '未設定'}")
    print(f"Endpoint: {ENDPOINT}")
    
    # システムプロンプト
    system_prompt = """
# Purpose
Your task is image editing. Based on the image provided by the user, generate a new image according to the user's instructions.

# Rules
* Objects unrelated to the user's instructions must remain completely identical to the original image.
* Edit only the elements specified by the user, and follow the instructions faithfully.
* If the user requests a **replacement**, replace the specified object in the original image with the new one.
* If the user requests a **removal**, erase the specified object from the original image.
* If the user requests an **addition**, add the specified object to the original image.
"""

    model_name = "google/gemini-2.5-flash-image-preview:free"

    # ==========　一度に生成する生成枚数の指定 ==========
    generate_images = 50

    # ========== 読み込む画像のパス ==========
    #file_path = "inputs/sample1.png"
    #file_path = "inputs/sample2.png"
    file_path = "inputs/images3.png"

    # ========== 編集内容の指定 ==========
    #query = "Turn it into a 3D figure. However, make the pose a cooler, more stylish girl's pose."
    #query = "Since this is a character from the 'asap' account, please add a cool 'asap' logo. Make sure it looks stylish and blends in naturally without feeling out of place."
    query = "Create a 4-panel manga comic strip featuring the girl from the image, showing her playing happily with friends"

    # =================================

    # 画像をbase64に変換
    file_b64 = convert_to_base64(file_path)
    print("ファイルのbase64変換が完了したので、処理を開始します。")

    for i in range(generate_images):
        print(f"\n画像{i+1}の生成を開始します。")
        
        # OpenRouter APIを直接呼び出し
        response = call_openrouter_api(
            model_name = model_name,
            system_prompt=system_prompt,
            user_query=query,
            image_base64=file_b64,
            api_key=API_KEY,
            endpoint=ENDPOINT,
            temperature=0.7
        )
        
        if response is None:
            print(f"画像{i+1}の生成に失敗しました。")
            continue
    

        # LLMの出力結果が正しく画像になっているかバリデーション
        # その後、画像を保存
        try:
            image_str_dict = validate_and_extract_base64(response)
            if image_str_dict:
                # 画像データが含まれているか確認
                has_image = any('base64' in item for item in image_str_dict)
                if has_image:
                    # 画像を保存、テキストは表示
                    process_dict_str_and_image(image_str_dict, file_path)
                else:
                    print("画像データが含まれていません。テキストのみのレスポンスです。")
                    # テキストのみ表示
                    for item in image_str_dict:
                        if 'str' in item:
                            print(f"レスポンス: {item['str']}")
            else:
                print("画像データが見つかりませんでした。")
                
        except Exception as e:
            print(f"エラー発生: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()