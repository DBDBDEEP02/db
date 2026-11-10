"""[심화2] AI 분리수거 도우미 - 채팅형 Streamlit 웹 서비스.

Claude 같은 채팅 화면에서 글 또는 사진으로 분리수거 방법을 물어볼 수 있습니다.
  - 글 질문  : "치킨 뼈는 음식물 쓰레기야?" -> 지식 사전 + 다국어 문장 임베딩(NLP)으로 품목 찾기
  - 사진 질문: 입력창의 + 버튼으로 사진 첨부 -> CLIP 제로샷 이미지 분류(CV)
  - 답변 전에 '생각 과정'(어떤 모델이 무엇을 찾았는지)을 접을 수 있는 형태로 보여주고,
    답변은 한 글자씩 흘러나오듯(스트리밍) 표시합니다.

실행 방법:  streamlit run app2.py
"""

import io
import time

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from recycle import RecycleClassifier
from recycle_chat import USAGE, Answer, TextMatcher, image_answer, item_answer, KB

BOT_AVATAR = ":material/recycling:"
SUGGESTIONS = {
    ":orange[:material/help:] 치킨 뼈는 음식물 쓰레기야?": "치킨 뼈는 음식물 쓰레기야?",
    ":blue[:material/local_drink:] 우유팩 어떻게 버려?": "우유팩 어떻게 버려?",
    ":green[:material/ac_unit:] 아이스팩은?": "아이스팩은 어디에 버려?",
    ":red[:material/medication:] 유통기한 지난 약": "유통기한 지난 약은 어떻게 버려?",
}

st.set_page_config(page_title="AI 분리수거 도우미", page_icon=BOT_AVATAR)


# ---------------------------------------------------------------------------
# 모델 관리: 캐싱 (모델 다운로드와 지식 사전 벡터 계산은 처음 한 번만)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="사진 분석 모델(CLIP)을 준비하는 중...")
def get_classifier() -> RecycleClassifier:
    return RecycleClassifier()


@st.cache_resource(show_spinner="한국어 이해 모델을 준비하는 중...")
def get_matcher() -> TextMatcher:
    return TextMatcher()


# ---------------------------------------------------------------------------
# 세션 상태 초기화
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    # 각 메시지: {"role", "content", "image"(PNG bytes 또는 None), "thinking"(list), "thinking_label"}
    st.session_state.messages = []
if "counts" not in st.session_state:
    st.session_state.counts = {}  # 분류별 질문 횟수 (사이드바 통계)
if "pending" not in st.session_state:
    st.session_state.pending = None  # 추천 질문을 눌렀을 때 보낼 문장
if "pending_photo" not in st.session_state:
    st.session_state.pending_photo = None  # 카메라로 찍어서 보낼 사진 (bytes)


def to_png(image: Image.Image, max_side: int = 480) -> bytes:
    """대화 기록에 저장할 사진을 적당한 크기의 PNG로 바꿉니다."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def stream_text(text: str):
    """답변을 단어 단위로 조금씩 내보내 타이핑하는 것처럼 보이게 합니다."""
    for i, word in enumerate(text.split(" ")):
        yield word if i == 0 else " " + word
        time.sleep(0.015)


def render_thinking(lines: list[str]) -> None:
    for line in lines:
        st.markdown(f"- {line}")


@st.dialog("카메라로 찍어서 물어보기")
def camera_dialog() -> None:
    """카메라 촬영 창. 찍은 사진(+ 함께 보낼 말)을 채팅으로 전송합니다."""
    shot = st.camera_input("쓰레기 하나가 화면 가운데에 크게 나오게 찍어 주세요.")
    note = st.text_input("함께 보낼 말 (선택)", placeholder="예: 이거 어디에 버려?")
    if st.button("이 사진으로 물어보기", icon=":material/send:", type="primary", disabled=shot is None):
        st.session_state.pending_photo = shot.getvalue()
        st.session_state.pending = note.strip() or None
        st.rerun()  # 창을 닫고 채팅에서 바로 답변


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.counts = {}
    st.session_state.pop("suggestion", None)  # 이전에 고른 추천 질문이 다시 전송되지 않게 초기화


# ---------------------------------------------------------------------------
# 사이드바: 서비스 소개 + 오늘의 질문 통계
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header(f"{BOT_AVATAR} 분리수거 도우미")
    st.caption("두 가지 AI 모델이 함께 일해요.")
    st.markdown(
        "- **사진** → OpenAI **CLIP** (제로샷 이미지 분류)\n"
        "- **글** → 다국어 **MiniLM** 문장 임베딩 + 분리수거 지식 사전 "
        f"({len(KB)}개 품목)"
    )
    st.button("새 대화 시작", icon=":material/add_comment:", on_click=reset_chat, width="stretch")

    st.subheader("오늘 물어본 쓰레기")
    if st.session_state.counts:
        counts = pd.DataFrame(
            {"분류": list(st.session_state.counts), "횟수": list(st.session_state.counts.values())}
        )
        st.bar_chart(counts, x="분류", y="횟수", horizontal=True, sort="-횟수", height=240,
                     alt="오늘 물어본 쓰레기 분류별 횟수 막대 차트")
    else:
        st.caption("아직 없어요.")
    st.caption("※ 일반적인 국내 기준이에요. 지자체마다 세부 기준이 다를 수 있어요.")


# ---------------------------------------------------------------------------
# 채팅 화면
# ---------------------------------------------------------------------------
st.title("무엇을 버리시나요?")

# 첫 화면: 인사말 + 추천 질문
if not st.session_state.messages:
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown("안녕하세요! 저는 **AI 분리수거 도우미**예요. 헷갈리는 쓰레기가 있으면 물어보세요.\n\n" + USAGE)
    picked = st.pills("추천 질문", list(SUGGESTIONS), label_visibility="collapsed", key="suggestion")
    if picked:
        st.session_state.pending = SUGGESTIONS[picked]

# 지난 대화 다시 그리기
for msg in st.session_state.messages:
    avatar = BOT_AVATAR if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("image"):
            st.image(msg["image"], width=240, alt="사용자가 첨부한 쓰레기 사진")
        if msg.get("thinking"):
            with st.expander(msg["thinking_label"], type="compact"):
                render_thinking(msg["thinking"])
        if msg["content"]:
            st.markdown(msg["content"])

# 카메라 버튼: 누르면 촬영 창이 열리고, 찍은 사진이 바로 채팅으로 전송됨
with st.container(horizontal=True, vertical_alignment="center"):
    if st.button("카메라로 찍기", icon=":material/photo_camera:"):
        camera_dialog()
    st.caption("또는 아래 입력창의 **+** 버튼으로 사진 파일을 첨부할 수 있어요. (휴대폰에서는 + 버튼에서 바로 촬영 가능)")

# 입력창: + 버튼으로 사진 첨부 가능
prompt = st.chat_input(
    "쓰레기 이름을 물어보거나 + 버튼으로 사진을 올려 보세요",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp"],
    submit_mode="disable",  # 답변하는 동안 입력창 잠시 비활성화
)

text, image = "", None
if prompt:
    text = (prompt.text or "").strip()
    image = Image.open(prompt.files[0]) if prompt.files else None
elif st.session_state.pending_photo:  # 카메라 창에서 찍은 사진
    image = Image.open(io.BytesIO(st.session_state.pending_photo))
    text = st.session_state.pending or ""
    st.session_state.pending_photo, st.session_state.pending = None, None
elif st.session_state.pending:
    text, st.session_state.pending = st.session_state.pending, None

if text or image is not None:
    # --- 사용자 메시지 ---------------------------------------------------------
    image_png = to_png(image) if image is not None else None
    with st.chat_message("user"):
        if image_png:
            st.image(image_png, width=240, alt="사용자가 첨부한 쓰레기 사진")
        if text:
            st.markdown(text)
    st.session_state.messages.append({"role": "user", "content": text, "image": image_png})

    # --- 도우미 답변 -----------------------------------------------------------
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        start = time.perf_counter()
        thinking: list[str] = []
        with st.status(":shimmer[생각하는 중...]", type="compact") as status:
            if image is not None:
                with st.status("사진 분석 (CLIP)", type="step", state="running") as step:
                    result = get_classifier().classify(image)
                    answer = image_answer(result)
                    render_thinking(answer.thinking)
                    step.update(state="complete")
                thinking += answer.thinking
                # 사진과 함께 글로 품목 이름도 말했다면, 글에 나온 품목 안내를 덧붙임
                if text:
                    matcher = get_matcher()
                    exact = matcher.find_exact(text)
                    if exact is not None:
                        extra = f"글에서 말씀하신 **{KB[exact].name}** 기준으로도 알려드릴게요."
                        answer = Answer(answer.text + "\n\n---\n\n" + extra + "\n\n" + item_answer(KB[exact]),
                                        answer.thinking, KB[exact].category)
                        thinking.append(f"글에서 품목 이름 '{KB[exact].name}'도 찾았어요.")
            else:
                with st.status("질문 이해하기 (문장 임베딩 + 지식 사전)", type="step", state="running") as step:
                    answer = get_matcher().answer(text)
                    render_thinking(answer.thinking)
                    step.update(state="complete")
                thinking += answer.thinking
            label = f"{time.perf_counter() - start:.1f}초 동안 생각했어요"
            status.update(label=label, state="complete")

        st.write_stream(stream_text(answer.text))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "image": None, "thinking": thinking, "thinking_label": label}
    )
    if answer.category:
        st.session_state.counts[answer.category] = st.session_state.counts.get(answer.category, 0) + 1
    st.rerun()  # 사이드바 통계와 추천 질문 영역을 최신 상태로 다시 그림
