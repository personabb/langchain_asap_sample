# Gemini 2.5 画像生成・画像編集

Google Gemini 2.5を使用した画像生成・画像編集のサンプルプロジェクトです。直接APIアプローチとLangChainアプローチの両方を提供しています。

[こちら](https://zenn.dev/asap)で記事を書いていますので、参照ください。

## 環境構築

### 前提条件
- Python 3.11以上
- uv (Pythonパッケージマネージャー)
- Google Cloudアカウントとプロジェクト

### セットアップ手順

1. **リポジトリのクローン**
   ```bash
   git clone <repository-url>
   cd gemini2.5_image_generation
   ```

2. **依存関係のインストール（uvを使用）**
   ```bash
   uv sync
   ```

3. **Google Cloud認証の設定**
   - Google Cloudプロジェクトでサービスアカウントを作成
   - サービスアカウントキーをJSON形式でダウンロード
   - `auth_json/auth.json` として配置
   - `.env`ファイルを作成し、以下を設定：
   ```env
   GOOGLE_APPLICATION_CREDENTIALS="auth_json/auth.json"
   ```

## スクリプト解説

### 1. vertexai_image_generation_api.py
Google GenAI APIを直接使用した画像生成スクリプトです。

#### 主要パラメータ
- `generate_images` (デフォルト: 1): 生成する画像枚数
- `query`: 画像生成のプロンプト内容
- `MODEL_ID`: 使用するモデル (gemini-2.5-flash-image-preview)

#### 実行方法
```bash
python vertexai_image_generation_api.py
```

### 2. vertexai_image_generation_langchain.py
LangChainを使用した画像生成スクリプトです。

#### 主要パラメータ
- `generate_images` (デフォルト: 1): 生成する画像枚数
- `query`: 画像生成のプロンプト内容
- `model_name`: 使用するモデル (gemini-2.5-flash-image-preview)

#### 実行方法
```bash
python vertexai_image_generation_langchain.py
```

### 3. vertexai_image_editing_api.py
Google GenAI APIを直接使用した画像編集スクリプトです。

#### 主要パラメータ
- `generate_images` (デフォルト: 1): 生成する画像枚数
- `file_path`: 編集元画像のパス (例: "inputs/sample.png")
- `query`: 画像編集の指示内容
- `MODEL_ID`: 使用するモデル (gemini-2.5-flash-image-preview)

#### 実行方法
1. 編集したい画像を`inputs/`ディレクトリに配置
2. スクリプト内の`file_path`と`query`を設定
3. 実行:
```bash
python vertexai_image_editing_api.py
```

### 4. vertexai_image_editing_langchain.py
LangChainを使用した画像編集スクリプトです。

#### 主要パラメータ
- `generate_images` (デフォルト: 1): 生成する画像枚数
- `file_path`: 編集元画像のパス
- `query`: 画像編集の指示内容
- `model_name`: 使用するモデル (gemini-2.5-flash-image-preview)

#### 実行方法
1. 編集したい画像を`inputs/`ディレクトリに配置
2. スクリプト内のパラメータを設定
3. 実行:
```bash
python vertexai_image_editing_langchain.py
```

## 出力ディレクトリ構造

- **画像生成**: `outputs/new_generation/generate_YYYY-MM-DD_HH-MM-SS.png`
- **画像編集**: `outputs/[元画像名]/[元画像名]_YYYY-MM-DD_HH-MM-SS.png`

## 技術仕様

- **使用モデル**: gemini-2.5-flash-image-preview
- **対応画像形式**: PNG, JPEG
- **認証方式**: Google Cloud Service Account

## 注意事項

- Google Cloud Vertex AIのAPI制限に注意してください 
- サービスアカウントキーは適切に管理し、公開リポジトリにコミットしないでください
- 大量の画像生成を行う場合は、RPM制限を考慮して`generate_images`パラメータを調整してください

## APIとLangChainの違い

- **API直接アプローチ**: Google GenAIクライアントを直接使用、より低レベルな制御が可能
- **LangChainアプローチ**: LangChainフレームワークを使用、プロンプト管理やチェーン化が容易

