"""손글씨 숫자 인식 Streamlit 웹 서비스.

화면 구성 (미션 가이드라인의 4개 영역)
  1. 입력 캔버스        : 마우스로 숫자를 그리는 영역 (streamlit-drawable-canvas)
  2. 전처리 이미지 표시 : 모델에 실제로 들어가는 28x28 이미지
  3. 모델 추론 결과     : 0~9 각 숫자의 예측 확률 막대 차트
  4. 이미지 저장소      : 저장한 그림과 예측 레이블/확률 목록

실행 방법:  streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from model import create_session, predict, preprocess

CANVAS_SIZE = 280  # 캔버스 크기(px). 28x28의 10배라서 전처리 결과와 비교하기 쉽습니다.

st.set_page_config(page_title="손글씨 숫자 인식", page_icon=":material/draw:", layout="wide")


# ---------------------------------------------------------------------------
# 모델 관리: 캐싱
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="MNIST ONNX 모델을 불러오는 중...")
def load_model():
    """모델을 한 번만 내려받고 불러옵니다.

    st.cache_resource 덕분에 화면이 다시 그려질 때(rerun)나 다른 사용자가
    접속했을 때(다른 세션)도 같은 추론 세션 객체를 재사용합니다.
    """
    return create_session()


# ---------------------------------------------------------------------------
# 세션 상태 초기화 (사용자마다 따로 유지되는 값)
# ---------------------------------------------------------------------------
if "gallery" not in st.session_state:
    st.session_state.gallery = []  # 저장한 이미지 목록 (이미지 저장소)
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0  # 값을 바꾸면 캔버스가 새로 만들어져 지워짐


def clear_canvas():
    st.session_state.canvas_key += 1


def to_png_bytes(image: Image.Image) -> bytes:
    """PIL 이미지를 PNG 바이트로 바꿉니다. (세션 상태에 가볍게 저장하기 위함)"""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 화면
# ---------------------------------------------------------------------------
st.title(":material/draw: 손글씨 숫자 인식")
st.caption("캔버스에 0부터 9까지 숫자 하나를 그려 보세요. MNIST ONNX 모델이 어떤 숫자인지 예측합니다.")

session = load_model()

col_canvas, col_processed, col_result = st.columns([1.1, 0.9, 1.4], gap="large")

# 1. 입력 캔버스 --------------------------------------------------------------
with col_canvas:
    st.subheader("1. 입력 캔버스")
    canvas = st_canvas(
        stroke_width=18,           # MNIST처럼 굵은 획
        stroke_color="#FFFFFF",    # 흰 글씨
        background_color="#000000",  # 검은 배경 (MNIST와 같은 형태)
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode="freedraw",
        return_image_data=True,    # 그린 그림을 numpy 배열로 받기 위해 필요
        key=f"canvas_{st.session_state.canvas_key}",
    )
    st.button("캔버스 지우기", icon=":material/ink_eraser:", on_click=clear_canvas)

# 캔버스 이미지 -> 전처리 -> 추론
image_rgba = canvas.image_data  # (H, W, 4) RGBA, 아직 그리지 않았으면 None
model_input = preprocess(image_rgba) if image_rgba is not None else None
probs = predict(session, model_input) if model_input is not None else None

# 2. 전처리 이미지 표시 -------------------------------------------------------
with col_processed:
    st.subheader("2. 전처리 이미지")
    if model_input is None:
        st.info("숫자를 그리면 모델 입력 이미지가 여기에 표시됩니다.", icon=":material/info:")
    else:
        # 28x28 이미지를 픽셀이 잘 보이도록 NEAREST 방식으로 10배 확대해서 표시
        small = Image.fromarray((model_input[0, 0] * 255).astype(np.uint8))
        st.image(
            small.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.NEAREST),
            caption="28×28 흑백, 0~1 정규화 (10배 확대 표시)",
            alt="모델에 입력되는 28x28 전처리 이미지",
        )

# 3. 모델 추론 결과 ------------------------------------------------------------
with col_result:
    st.subheader("3. 모델 추론 결과")
    if probs is None:
        st.info("숫자를 그리면 0~9 각 숫자의 예측 확률이 표시됩니다.", icon=":material/info:")
    else:
        pred_label = int(np.argmax(probs))
        pred_prob = float(probs[pred_label])
        st.metric("예측 숫자", pred_label, f"확률 {pred_prob:.1%}", delta_color="off")

        chart_df = pd.DataFrame({"숫자": [str(i) for i in range(10)], "확률": probs})
        st.bar_chart(
            chart_df,
            x="숫자",
            y="확률",
            height=260,
            alt="0부터 9까지 각 숫자의 예측 확률 막대 차트",
        )

        # 현재 그림을 이미지 저장소에 추가
        if st.button("이미지 저장소에 저장", icon=":material/save:", type="primary"):
            thumb = Image.fromarray(image_rgba.astype(np.uint8)).convert("RGB")
            thumb = thumb.resize((140, 140), Image.Resampling.LANCZOS)
            st.session_state.gallery.append(
                {"image": to_png_bytes(thumb), "label": pred_label, "prob": pred_prob}
            )
            st.toast(f"숫자 {pred_label} 이미지를 저장했습니다.", icon=":material/check:")

# 4. 이미지 저장소 ------------------------------------------------------------
st.divider()
header_col, clear_col = st.columns([4, 1], vertical_alignment="center")
with header_col:
    st.subheader(f"4. 이미지 저장소 ({len(st.session_state.gallery)}개)")
with clear_col:
    if st.session_state.gallery and st.button("저장소 비우기", icon=":material/delete:"):
        st.session_state.gallery = []
        st.rerun()

if not st.session_state.gallery:
    st.caption("저장한 이미지가 없습니다. 숫자를 그린 뒤 '이미지 저장소에 저장'을 눌러 보세요.")
else:
    # 최근에 저장한 이미지가 앞에 오도록 역순으로, 한 줄에 6개씩 표시
    items = list(reversed(st.session_state.gallery))
    per_row = 6
    for start in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, item in zip(cols, items[start : start + per_row]):
            with col:
                st.image(
                    item["image"],
                    caption=f"예측: {item['label']} ({item['prob']:.1%})",
                    alt=f"저장된 손글씨 이미지, 예측 {item['label']}",
                )
