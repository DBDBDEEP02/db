import json

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="지구별 OX 퀴즈 정답 찾기", page_icon="🌍")

# OX 퀴즈 데이터: (문제, 정답) — 여기에 계속 추가하면 됨
quiz = [
    ("뱀은 뒤로도 기어갈 수 있다", "X"),
    ("미국 루스벨트 대통령은 경제 대공황을 해결하기 위해 뉴딜정책을 추진하였다", "O"),
    ("Z9별에서는 커플을 맺고 절대 헤어질 수 없다", "X"),
    ("데스 사이드는 대전무기이며 3번 무기슬롯에 장착 가능하다", "X"),
    ("오페라 \"라 보엠\"의 4막에서는 미미가 죽음을 맞이한다", "O"),
    ("산업혁명의 원동력이 된 증기기관을 최초 발명한 사람은 제임스 와트이다", "X"),
    ("칼륨은 원소 기호 C로 표기한다", "X"),
    ("제 1회 하계 올림픽은 그리스에서 개최되었다", "O"),
    ("농구 경기에서 자유투를 던져 넣을 경우에는 한 골당 3점을 부여한다", "X"),
    ("은으로 제작된 액세서리가 변색되었을 때는 레몬을 이용하면 색이 돌아온다", "O"),
    ("로베스 피에르는 영국 출신의 왕정 반대 혁명가이다", "X"),
    ("야구 경기에서 한 회는 15분으로 제한된다", "X"),
    ("현재 오팔아양에게서 의상을 씰로 변환할 수 없다", "O"),
    ("문익점은 원나라에 갔다가 붓대 속에 삼나무 씨앗을 넣어서 가져왔다", "X"),
    ("관객에게 눈물을 흘리게 해 감동과 카타르시스를 주는 영화를 신파 영화라 한다", "O"),
]

st.title("🌍 지구별 OX 퀴즈 정답 찾기")

# 입력할 때마다 바로 결과가 나오도록 검색은 브라우저(JavaScript)에서 처리
html = """
<style>
  body { font-family: sans-serif; margin: 0; }
  input { width: 100%; box-sizing: border-box; padding: 10px; font-size: 16px;
          border: 1px solid #ccc; border-radius: 8px; }
  .item { margin-top: 10px; padding: 12px; border-radius: 8px; font-size: 15px; }
  .O { background: #dff5e3; color: #1b6e2d; }
  .X { background: #fde2e2; color: #a12020; }
  .msg { margin-top: 10px; padding: 12px; border-radius: 8px; background: #fff4d6; color: #7a5b00; }
</style>
<input id="q" placeholder="문제 검색 (두 글자 이상) 예: 뱀은" autofocus>
<div id="out"></div>
<script>
  const quiz = QUIZ_DATA;
  const norm = s => s.replace(/ /g, "");
  const q = document.getElementById("q"), out = document.getElementById("out");
  q.addEventListener("input", () => {
    const k = norm(q.value);
    out.innerHTML = "";
    if (!k) return;
    if (k.length < 2) { out.innerHTML = '<div class="msg">두 글자 이상 입력해 주세요.</div>'; return; }
    const hits = quiz.filter(([text]) => norm(text).includes(k));
    if (!hits.length) { out.innerHTML = '<div class="msg">일치하는 문제가 없어요.</div>'; return; }
    for (const [text, ans] of hits) {
      const d = document.createElement("div");
      d.className = "item " + ans;
      d.textContent = text + "  →  " + (ans === "O" ? "⭕ O" : "❌ X");
      out.appendChild(d);
    }
  });
</script>
""".replace("QUIZ_DATA", json.dumps(quiz, ensure_ascii=False))

components.html(html, height=600, scrolling=True)
