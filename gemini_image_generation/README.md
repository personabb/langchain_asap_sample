# LangChainでgeminiの画像生成・画像編集

[こちら](https://zenn.dev/asap)で記事を書いていますので、参照ください。

## スクリプト解説

### gemini_image_generation.py
Google Gemini APIを使用して画像を生成するスクリプトです。

#### 主要パラメータ
- `generate_images` (デフォルト: 5): 生成する画像枚数 (APIのRPM制限10に注意)
- `query`: 画像生成のプロンプト内容 (例: "家のPCデスクの実写画像を作成してください...")
- `model`: 使用するGeminiモデル (デフォルト: "models/gemini-2.0-flash-exp-image-generation")、現状はこのモデルしか対応していません。

#### 実行方法
1. `.env`ファイルにGoogle APIキーを設定
2. スクリプトを実行: `python gemini_image_generation.py`
3. 生成画像は`outputs/new_generation/`に保存されます

### gemini_image_editing.py
既存画像を編集するスクリプトです。

#### 主要パラメータ
- `generate_images` (デフォルト: 5): 生成する画像枚数 (APIのRPM制限10に注意)
- `file_path`: 編集元画像のパス (例: "inputs/sample1.png")
- `query`: 画像編集の指示内容 (例: "表情を笑顔に変更して")
- `model`: 使用するGeminiモデル (デフォルト: "models/gemini-2.0-flash-exp-image-generation")、現状はこのモデルしか対応していません。

#### 実行方法
1. `.env`ファイルにGoogle APIキーを設定
2. 編集したい画像を`inputs/`ディレクトリに配置
3. スクリプトを実行: `python gemini_image_editing.py`
4. 編集画像は`outputs/[元画像名]/`に保存されます

## 注意事項
- Google APIの利用制限に注意してください (RPM: 10)

