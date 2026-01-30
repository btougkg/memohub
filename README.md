
# 🧠 MemoHub

ローカルで「メモ・URL・スニペット」を高速・重複防止で管理できる**軽量メモ管理アプリ**。

## 主な特徴

- SQLite（単一ファイルDB）で保存、セットアップ不要
- URLは自動判定＆正規化（canonicalize）＋UNIQUE制約で重複防止
- タイトル・内容・説明・タグ・URLでキーワード検索
- 意味検索（セマンティック検索/embedding）で類似候補をAI提案
- 追加時に重複/類似があれば上書きor新規登録を選択
- 編集・削除もワンクリック
- 1000件以下の用途に最適化

---

## セットアップ（3分）

### 1. 必要なパッケージをインストール

```bash
pip install streamlit sentence-transformers numpy
```

または

```bash
pip install -r requirements.txt
```

### 2. 起動

```bash
streamlit run memohub.py
```

ブラウザで http://localhost:8501 が自動で開きます。

---

## 使い方

### 🔎 検索
- タイトル・内容・説明・タグ・URLでキーワード検索
- URL入力時は正規化URLで完全一致があれば先に提示
- セマンティック検索（AI意味検索）もON/OFF可

### ➕ 追加
- 「登録」で重複/類似チェック
- 類似なし→自動登録、類似/重複あり→候補提示後に上書き/新規/キャンセルを選択

### ✏️ 編集
- 検索結果カード右上の✏️ボタンで編集画面へ
- 編集完了後は検索画面に戻る

### 🗑️ 削除
- 検索結果カード右上の🗑️ボタンで削除

---

## 依存パッケージ

- streamlit
- sentence-transformers（AI意味検索用、なくても動作可）
- numpy

requirements.txt例：
```txt
streamlit
sentence-transformers
numpy
```

---

## 環境変数（任意）

| 変数                    | 意味             | 例                                                            |
| ----------------------- | ---------------- | ------------------------------------------------------------- |
| `MEMOHUB_DB`            | DBファイルパス   | `memohub.db`                                                  |
| `MEMOHUB_EMBED_MODEL`   | embeddingモデル  | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `MEMOHUB_SIM_THRESHOLD` | 類似判定しきい値 | `0.86`                                                        |

例：
```bash
export MEMOHUB_DB=./data/memohub.db
export MEMOHUB_SIM_THRESHOLD=0.84
streamlit run memohub.py
```

---

## トラブルシューティング

### 検索が遅い
- 1000件以下なら高速
- 追加/編集直後にサイドバー「ベクトル索引を再読込」で再構築

### AIモデルのダウンロードが遅い
- 初回のみ数百MBダウンロード、2回目以降は即起動

### huggingface.co タイムアウト
- `HF_HUB_READ_TIMEOUT` / `HF_HUB_ETAG_TIMEOUT` を60以上に設定

---

## データ移行

既存の memohub.db から移行したい場合は、スキーマが異なる場合は簡単な移行スクリプトで対応可能です。

---

## ライセンス

MIT License - 自由に使用・改変可能
