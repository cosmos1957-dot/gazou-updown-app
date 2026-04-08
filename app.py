import math
import time
from io import BytesIO
from pathlib import Path
from typing import Tuple
import urllib.error
import urllib.request

import numpy as np
import streamlit as st
import cv2
from PIL import Image


DEFAULT_SCALE = 4
DEFAULT_TILE = 0  # FSRCNNは軽量なため、既定はタイル分割なし
MAX_PIXELS = 100_000_000  # 100MP
TARGET_MAX_PIXELS = 99_000_000  # 99MPに収める（安全側）

# --- Upscaler backend: OpenCV dnn_superres (FSRCNN x4) ---
# pipで扱いやすいOpenCVの事前学習済みモデルを利用する。
OPENCV_DNNSR_MODEL_NAME = "FSRCNN_x4.pb"
OPENCV_DNNSR_ALGO = "fsrcnn"
MODEL_URLS = [
    "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x4.pb",
    "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb",
]


def _pillow_lanczos_resample() -> int:
    # Pillowのバージョン差分吸収（古い環境でも動かすため）
    try:
        return Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    except AttributeError:
        return Image.LANCZOS


def _composite_to_rgb(pil_img: Image.Image, bg: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    OpenCV DNN SuperResへ渡す前に、RGBA等は背景に合成してRGB化する。
    """
    if pil_img.mode == "RGB":
        return pil_img
    if pil_img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", pil_img.size, bg)
        background.paste(pil_img.convert("RGBA"), mask=pil_img.convert("RGBA").split()[-1])
        return background
    # それ以外（例: L / P）をRGBへ
    return pil_img.convert("RGB")


def resize_by_long_side(pil_img: Image.Image, long_side_px: int) -> Image.Image:
    w, h = pil_img.size
    long_side_px = int(long_side_px)
    if long_side_px <= 0:
        raise ValueError("long_side_px must be a positive integer")

    current_long = max(w, h)
    if current_long == 0:
        raise ValueError("Invalid image size")

    scale = long_side_px / current_long
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return pil_img.resize((new_w, new_h), resample=_pillow_lanczos_resample())


def enforce_under_max_pixels(pil_img: Image.Image, st_module=st) -> Image.Image:
    """
    Pillowで保存する前に100MPを超えないようにする。
    超える場合は警告し、99MP相当に縮小して返す。
    """
    warned = False
    while True:
        w, h = pil_img.size
        pixels = w * h
        if pixels <= MAX_PIXELS:
            return pil_img

        if not warned:
            warned = True
            st_module.warning(
                f"出力が100MPを超えそうです（現在: {pixels:,}px）。"
                " Adobe Stock対策として99MP相当に自動縮小します。"
            )

        scale = math.sqrt(TARGET_MAX_PIXELS / float(pixels))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        # 丸めの影響でサイズが変わらない場合があるため、安全側に縮める
        if new_w == w and new_h == h:
            if w >= h:
                new_w = max(1, w - 1)
            else:
                new_h = max(1, h - 1)
        pil_img = pil_img.resize((new_w, new_h), resample=_pillow_lanczos_resample())


def pil_to_bgr_np(pil_img: Image.Image) -> np.ndarray:
    """
    Pillow(RGB) -> numpy(BGR, uint8)
    OpenCV DNN SuperResはOpenCV風のBGR配列を想定する。
    """
    # OpenCVを依存に入れた方が確実だが、ここでは cv2 を直接使わないため簡易変換
    # RGB -> BGR はチャネル順を入れ替えるだけでOK（uint8想定）
    rgb = np.array(pil_img, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB image converted to shape (H, W, 3)")
    bgr = rgb[:, :, ::-1].copy()
    return bgr


def bgr_np_to_pil(bgr: np.ndarray) -> Image.Image:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected BGR image array shape (H, W, 3)")
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def ensure_model_exists(model_path: Path, st_module=st) -> Path:
    """
    FSRCNNモデルがなければ自動ダウンロードする。
    GitHubのRaw URLを優先し、失敗時はフォールバックURLを試す。
    """
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, url in enumerate(MODEL_URLS, start=1):
        try:
            st_module.info(f"モデルを自動取得中... ({idx}/{len(MODEL_URLS)})")
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            if not data:
                raise ValueError("Downloaded model file is empty.")
            model_path.write_bytes(data)
            return model_path
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            continue

    raise FileNotFoundError(
        "FSRCNN model download failed. "
        "Please place models/FSRCNN_x4.pb manually and retry."
    )


@st.cache_resource(show_spinner=False)
def load_upsampler(tile: int = DEFAULT_TILE):
    """
    OpenCV dnn_superres の推論器をロードする（初回のみ重い）。
    GPUが使えるOpenCVビルドの場合はCUDAを試み、無理ならCPUにフォールバックする。

    NOTE:
    - OpenCVのpipホイールは通常CUDA非対応なので、多くの環境ではCPU動作になります。
    - tile引数は、ここでは「タイル分割でOOM/メモリ負荷を抑える」目的で使用します（推論器の設定ではなく上位ロジックで利用）。
    """
    app_dir = Path(__file__).resolve().parent
    model_dir = app_dir / "models"
    model_path = model_dir / OPENCV_DNNSR_MODEL_NAME

    model_path = ensure_model_exists(model_path, st_module=st)

    # OpenCV dnn_superres
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel(OPENCV_DNNSR_ALGO, DEFAULT_SCALE)

    # 可能ならCUDAバックエンドを試す（OpenCVがCUDA有効ビルドの場合のみ）
    try:
        net = sr.getNet()
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        backend = "cuda"
    except Exception:
        backend = "cpu"

    return {"sr": sr, "backend": backend, "tile": int(tile)}


def upscale_4x(pil_img: Image.Image, tile: int = DEFAULT_TILE) -> Image.Image:
    bundle = load_upsampler(tile=tile)
    sr = bundle["sr"]

    # OpenCVはBGR uint8
    rgb_img = _composite_to_rgb(pil_img)
    bgr = pil_to_bgr_np(rgb_img)

    # tile=0なら一括処理
    tile_size = int(tile)
    if tile_size <= 0:
        out_bgr = sr.upsample(bgr)
        return bgr_np_to_pil(out_bgr)

    # タイル分割でメモリ負荷を抑える（境界の継ぎ目対策にオーバーラップ）
    h, w = bgr.shape[:2]
    pad = 10  # オーバーラップ（tile_pad相当の役割）

    out_h, out_w = h * DEFAULT_SCALE, w * DEFAULT_SCALE
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # 走査
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)

            # オーバーラップ込みの切り出し
            ys0 = max(0, y0 - pad)
            xs0 = max(0, x0 - pad)
            ys1 = min(h, y1 + pad)
            xs1 = min(w, x1 + pad)
            tile_bgr = bgr[ys0:ys1, xs0:xs1]

            tile_up = sr.upsample(tile_bgr)

            # オーバーラップ部分をトリミングして貼り戻す
            trim_top = (y0 - ys0) * DEFAULT_SCALE
            trim_left = (x0 - xs0) * DEFAULT_SCALE
            trim_bottom = (ys1 - y1) * DEFAULT_SCALE
            trim_right = (xs1 - x1) * DEFAULT_SCALE

            th, tw = tile_up.shape[:2]
            y_start = trim_top
            y_end = th - trim_bottom if trim_bottom > 0 else th
            x_start = trim_left
            x_end = tw - trim_right if trim_right > 0 else tw

            core = tile_up[y_start:y_end, x_start:x_end]
            oy0 = y0 * DEFAULT_SCALE
            ox0 = x0 * DEFAULT_SCALE
            oy1 = oy0 + core.shape[0]
            ox1 = ox0 + core.shape[1]
            out[oy0:oy1, ox0:ox1] = core

    return bgr_np_to_pil(out)


def _image_bytes_to_pil(uploaded_file) -> Image.Image:
    uploaded_file.seek(0)
    pil_img = Image.open(uploaded_file)
    # Exif回転などが必要ならここで対応できるが、今回は最小構成で実装する
    return pil_img


def main():
    st.set_page_config(page_title="FSRCNN 4x + Long Side Resize", layout="wide")
    st.title("4x アップスケール + Long Side縮小（Streamlit）")
    st.write(
        "OpenCV DNN SuperRes（FSRCNN x4）で4倍アップスケールした後、PillowでLong Side指定のピクセルに縮小します。"
    )

    with st.sidebar:
        st.header("設定")

        uploaded = st.file_uploader(
            "画像をアップロード",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        )

        output_long_side = st.number_input(
            "出力: Long Side（px）",
            min_value=1,
            max_value=20000,
            value=2048,
            step=1,
        )

        # FSRCNNは軽量なので、既定はタイルなし（0）で高速処理を優先
        tile = st.number_input(
            "tile（必要時のみ分割）",
            min_value=0,
            max_value=8000,
            value=DEFAULT_TILE,
            step=1,
            help="0でタイル処理なし（推奨）。メモリ不足時のみ大きめ値で分割してください。",
        )

        output_format = st.selectbox("保存形式", options=["png", "jpeg"], index=0)
        jpeg_quality = st.slider("JPEG品質", min_value=10, max_value=95, value=92, step=1)

        run = st.button("変換（4x → Long Side縮小）", type="primary", disabled=(uploaded is None))

    if uploaded is None:
        st.info("画像をアップロードすると、ここに入力プレビューが表示されます。")
        return

    # 入力プレビュー
    input_pil = _image_bytes_to_pil(uploaded)
    input_pil_rgb = _composite_to_rgb(input_pil)
    st.subheader("入力")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(input_pil_rgb, use_column_width=True)
    with col2:
        w, h = input_pil_rgb.size
        st.metric("入力ピクセル", f"{w} x {h}")

    if not run:
        return

    # 処理開始
    start = time.time()
    with st.spinner("4倍アップスケール中..."):
        upscaled = upscale_4x(input_pil, tile=int(tile))

    with st.spinner("PillowでLong Side指定に縮小中..."):
        resized = resize_by_long_side(upscaled, int(output_long_side))

    # 保存前チェック（100MPガード）
    resized = enforce_under_max_pixels(resized, st_module=st)

    # 出力プレビュー
    st.subheader("出力")
    st.image(resized, use_column_width=True)

    w2, h2 = resized.size
    elapsed = time.time() - start
    st.success(f"完了: {w2} x {h2}（約 {elapsed:.1f} 秒）")

    # DL用バイト列
    buf = BytesIO()
    if output_format == "png":
        resized.save(buf, format="PNG")
        mime = "image/png"
        ext = "png"
    else:
        resized.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
        mime = "image/jpeg"
        ext = "jpg"

    buf.seek(0)
    st.download_button(
        label=f"ダウンロード（.{ext}）",
        data=buf,
        file_name=f"upscale4x_longside_{output_long_side}px.{ext}",
        mime=mime,
    )


if __name__ == "__main__":
    main()

