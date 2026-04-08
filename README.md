# FSRCNN 4x Upscale + Long Side Resize

## English

This Streamlit app upscales an image by **4x** using OpenCV DNN Super Resolution (**FSRCNN**) and then resizes it to your target **long side (px)** while keeping aspect ratio.

### Features
- 4x super-resolution with FSRCNN (`cv2.dnn_superres`)
- Long-side resize after upscaling
- **100MP guard** for stock workflows (for example Adobe Stock):  
  if output exceeds 100 megapixels, it is auto-resized to about 99MP
- PNG / JPEG export

### Model download behavior
On startup, if `models/FSRCNN_x4.pb` is missing, the app automatically downloads it from a GitHub raw URL fallback list.

### Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Community Cloud deploy
1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app** and select your repository.
4. Set:
   - Main file path: `app.py`
5. Deploy.

---

## 日本語

このStreamlitアプリは、OpenCV DNN Super Resolution（**FSRCNN**）で画像を**4倍拡大**し、その後にアスペクト比を保ったまま**長辺指定（px）**へ縮小します。

### 主な機能
- FSRCNNによる4倍アップスケール（`cv2.dnn_superres`）
- 拡大後のLong Side指定リサイズ
- ストック投稿向けの**100MPガード**（Adobe Stock想定）  
  出力が100MPを超える場合、自動で約99MPへ縮小
- PNG / JPEG で保存可能

### モデルの自動取得
起動時に `models/FSRCNN_x4.pb` が無い場合、GitHub Raw URLのフォールバック先から自動ダウンロードします。

### ローカル起動
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Community Cloud公開手順
1. このリポジトリをGitHubにpush
2. [Streamlit Community Cloud](https://share.streamlit.io/) にログイン
3. **New app** から対象リポジトリを選択
4. `Main file path` に `app.py` を指定
5. Deployで公開

