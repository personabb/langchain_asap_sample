# 【Playwright MCP】を利用したエージェントを【LangGraph】で構築する

[こちら](https://zenn.dev/asap)で記事を書いていますので、参照ください。

## スクリプト解説

### praywrite_mcp_langchain_tools.py

## 実行方法

### 前提条件
- Google AI StudioのAPIキーを取得済みであること
- MCPサーバーが設定済みであること(PlayWrite MCPが利用できること)

### 環境設定
1. `.env`ファイルを作成し、Google AI StudioのAPIキーを設定:
```
GOOGLE_APIKEY=your_api_key_here
```

2. `mcp_config.json`にMCPサーバーの設定を記述

### 依存関係のインストール
```bash
pip install -r requirements.txt
```

### スクリプトの実行
```bash
python praywrite_mcp_langchain_tools.py
```

### 使用方法
1. スクリプトを実行すると対話型プロンプトが表示されます
2. 質問を入力してEnterキーを押すと、AIが回答を生成します（改行はできませんので注意してください）
3. 終了する場合は「exit」または「quit」と入力
