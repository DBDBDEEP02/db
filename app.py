import streamlit as st

st.set_page_config(page_title="AI Prototype App", layout="wide")

st.sidebar.title("📚 메뉴")
st.sidebar.markdown("""
- 홈
- 예측
- 시각화
""")

st.title("AI Prototype — Streamlit Demo")
st.markdown("""
이 애플리케이션은 **AI 모델 예측과 시각화를 통합한 프로토타입**입니다.  
왼쪽 메뉴에서 원하는 기능을 선택하여 실습을 진행하세요.
""")

st.info("버전: v1.0-app")