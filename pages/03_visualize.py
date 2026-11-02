import streamlit as st
import altair as alt
from src.utils import load_log

st.header("예측 결과 시각화")

df = load_log()

if df.empty:
    st.warning("아직 예측 데이터가 없습니다.")
else:
    st.dataframe(df.tail(10))
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="text:N",
            y="prob:Q",
            color=alt.condition("datum.label == 1", alt.value("#2D6CDF"), alt.value("#FF4B4B"))
        )
        .properties(width=600, height=300)
    )
    st.altair_chart(chart, use_container_width=True)