"""MNIST ONNX 모델 다운로드, 이미지 전처리, 추론을 담당하는 모듈.

Streamlit 화면 코드(app.py)와 분리해 두어서
- 화면 없이도 단독으로 테스트할 수 있고
- Docker 빌드 단계에서 모델을 미리 내려받을 때도 재사용할 수 있습니다.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# ONNX 공식 모델 저장소(github.com/onnx/models)의 MNIST 모델 (opset 12)
MODEL_URL = (
    "https://github.com/onnx/models/raw/main/"
    "validated/vision/classification/mnist/model/mnist-12.onnx"
)
# 모델 파일은 이 소스 파일과 같은 폴더에 저장합니다.
MODEL_PATH = Path(__file__).resolve().parent / "mnist-12.onnx"

# 정상 모델 파일은 약 26KB입니다. Git LFS 포인터(텍스트, 약 130B)를
# 잘못 받은 경우를 걸러내기 위한 최소 크기입니다.
_MIN_MODEL_BYTES = 10_000


def download_model(path: Path = MODEL_PATH, url: str = MODEL_URL) -> Path:
    """모델 파일이 없으면 GitHub에서 내려받고, 파일 경로를 반환합니다."""
    if path.exists() and path.stat().st_size >= _MIN_MODEL_BYTES:
        return path  # 이미 받아 둔 파일이 있으면 다시 받지 않음

    tmp_path = path.with_suffix(".download")
    urllib.request.urlretrieve(url, tmp_path)
    if tmp_path.stat().st_size < _MIN_MODEL_BYTES:
        tmp_path.unlink()
        raise RuntimeError(f"모델 파일이 올바르지 않습니다: {url}")
    tmp_path.replace(path)  # 다운로드가 끝난 뒤에만 실제 이름으로 바꿈
    return path


def create_session(path: Path = MODEL_PATH) -> ort.InferenceSession:
    """모델을 내려받고(필요 시) ONNX Runtime 추론 세션을 만듭니다."""
    download_model(path)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def preprocess(image_rgba: np.ndarray) -> np.ndarray | None:
    """캔버스 이미지를 MNIST 모델 입력 형식으로 바꿉니다.

    입력: 캔버스에서 받은 (H, W, 4) RGBA 배열 (검은 배경 + 흰 글씨)
    출력: (1, 1, 28, 28) float32 배열, 값 범위 0~1
          아무것도 그리지 않았으면 None

    MNIST 학습 데이터와 같은 방식으로 맞춰 주면 인식률이 크게 올라갑니다.
      1) 흑백 변환
      2) 숫자가 있는 영역만 잘라내기
      3) 비율을 유지하며 20x20 안에 들어가게 줄이기
      4) 28x28 검은 바탕의 가운데(무게중심 기준)에 붙이기
      5) 0~1로 정규화하고 (1, 1, 28, 28)로 모양 바꾸기
    """
    # 1) RGBA -> 흑백(L). 배경이 검정, 글씨가 흰색이므로 밝을수록 글씨입니다.
    gray = Image.fromarray(image_rgba.astype(np.uint8)).convert("L")
    arr = np.asarray(gray)

    # 2) 글씨가 있는 픽셀(밝기 > 0)의 경계 상자를 찾아 잘라냅니다.
    ys, xs = np.nonzero(arr > 0)
    if len(xs) == 0:
        return None  # 빈 캔버스
    digit = gray.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    # 3) 긴 변이 20픽셀이 되도록 비율을 유지하며 축소합니다.
    w, h = digit.size
    scale = 20.0 / max(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    digit = digit.resize(new_size, Image.Resampling.LANCZOS)

    # 4) 28x28 검은 캔버스에 붙입니다. 먼저 가운데에 놓은 뒤,
    #    숫자의 무게중심이 정중앙(14, 14)에 오도록 위치를 보정합니다.
    canvas = Image.new("L", (28, 28), 0)
    left = (28 - new_size[0]) // 2
    top = (28 - new_size[1]) // 2
    canvas.paste(digit, (left, top))

    pixels = np.asarray(canvas, dtype=np.float32)
    total = pixels.sum()
    if total > 0:
        rows, cols = np.indices(pixels.shape)
        cy = (rows * pixels).sum() / total
        cx = (cols * pixels).sum() / total
        # 무게중심 보정으로 숫자가 28x28 밖으로 밀려나 잘리지 않도록,
        # 가장자리에 최소 2픽셀 여백을 남기는 범위 안으로 위치를 제한합니다.
        margin = 2
        new_left = min(max(left + int(round(14 - cx)), margin), 28 - margin - new_size[0])
        new_top = min(max(top + int(round(14 - cy)), margin), 28 - margin - new_size[1])
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(digit, (new_left, new_top))
        pixels = np.asarray(canvas, dtype=np.float32)

    # 5) 0~255 -> 0~1, (28, 28) -> (1, 1, 28, 28) [배치, 채널, 높이, 너비]
    return (pixels / 255.0).reshape(1, 1, 28, 28)


def softmax(logits: np.ndarray) -> np.ndarray:
    """모델 출력 점수(logit)를 합이 1인 확률로 바꿉니다."""
    exp = np.exp(logits - logits.max())  # 큰 값에서 overflow 방지
    return exp / exp.sum()


def predict(session: ort.InferenceSession, model_input: np.ndarray) -> np.ndarray:
    """전처리된 입력으로 추론하고, 0~9 각 숫자의 확률(길이 10 배열)을 반환합니다."""
    input_name = session.get_inputs()[0].name  # mnist-12 모델은 "Input3"
    logits = session.run(None, {input_name: model_input})[0]  # (1, 10)
    return softmax(logits[0])
