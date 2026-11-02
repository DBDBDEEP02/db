import csv
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="지구별 OX 퀴즈 정답 찾기", page_icon="🌍")

# OX 퀴즈 데이터는 data/quiz.csv (question,answer) — 여기에 계속 추가하면 됨
# 출처: https://blog.naver.com/dpdk1218/221795137191
with open(Path(__file__).parent / "data" / "quiz.csv", encoding="utf-8") as f:
    quiz = [(row["question"], row["answer"]) for row in csv.DictReader(f)]

st.title("🌍 지구별 OX 퀴즈 정답 찾기")
st.caption(f"문제 {len(quiz)}개")

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
<input id="q" placeholder="문제 검색 (두 글자 이상) 예: 뱀은, ㅂㅇ" autofocus>
<div id="out"></div>
<script>
  const quiz = QUIZ_DATA;
  const norm = s => s.replace(/ /g, "").toLowerCase();
  // 초성 검색: 입력한 글자가 초성(ㄱ~ㅎ)이면 문제 글자의 초성과 비교
  const CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ";
  const cho = c => {
    const n = c.charCodeAt(0) - 0xAC00;
    return n >= 0 && n < 11172 ? CHO[Math.floor(n / 588)] : c;
  };
  const matches = (text, k) => {
    for (let i = 0; i + k.length <= text.length; i++) {
      let ok = true;
      for (let j = 0; j < k.length && ok; j++) {
        const t = text[i + j];
        ok = k[j] === t || (CHO.includes(k[j]) && cho(t) === k[j]);
      }
      if (ok) return true;
    }
    return false;
  };
  const q = document.getElementById("q"), out = document.getElementById("out");
  q.addEventListener("input", () => {
    const k = norm(q.value);
    out.innerHTML = "";
    if (!k) return;
    if (k.length < 2) { out.innerHTML = '<div class="msg">두 글자 이상 입력해 주세요.</div>'; return; }
    const hits = quiz.filter(([text]) => matches(norm(text), k));
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
