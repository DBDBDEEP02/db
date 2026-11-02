import streamlit as st
from src.model import predict_texts
import pandas as pd
import time

st.header("🔮 텍스트 예측")             # 페이지 상단 헤더 출력

text = st.text_input("문장 입력", "good")  # 사용자로부터 예측할 문장을 입력받음 (기본값: "good")

if st.button("예측 실행"):              # 사용자가 버튼을 클릭했을 때만 아래 코드 실행
    t0 = time.perf_counter()           # 예측 시작 시점의 고해상도 시간 기록
    df = predict_texts([text])         # 입력 문장을 리스트로 감싸 모델에 전달하여 예측 수행
    dt = time.perf_counter() - t0      # 예측 처리에 걸린 총 시간 계산

    label = int(df["label"][0])        # 예측 결과 라벨을 정수형으로 추출
    prob = float(df["prob"][0])         # 예측 확률 값을 실수형으로 추출

    st.metric(label="예측 결과", value=f"{label}", delta=f"{prob*100:.1f}% 확신")  # 예측 라벨과 확신도(%)를 지표 형태로 표시
    st.toast("예측 완료! 🎉")           # 예측이 끝났음을 알리는 토스트 알림 표시
    st.info(f"처리 시간: {dt:.3f}s")    # 예측 처리 시간을 정보 메시지로 출력

    df["timestamp"] = pd.Timestamp.now()  # 예측 시점을 타임스탬프로 데이터프레임에 추가
    df.to_csv("data/log.csv", mode="a", header=False, index=False)  # 예측 결과를 로그 CSV 파일에 누적 저장