# 미션17 손글씨 숫자 인식 웹 서비스 (1팀 박단비)

캔버스에 숫자를 그리면 MNIST ONNX 모델이 0~9 중 어떤 숫자인지 예측하는 Streamlit 서비스입니다.

- Docker Hub: https://hub.docker.com/r/dbdbdeep02/mnist-streamlit

## 실행 방법

**Docker로 실행**
```bash
docker run -p 8501:8501 dbdbdeep02/mnist-streamlit:latest
```
브라우저에서 http://localhost:8501 접속

**로컬에서 실행** (Python 3.10)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 파일 구성
| 파일 | 설명 |
|---|---|
| `app.py` | Streamlit 화면 (입력 캔버스 / 전처리 이미지 / 추론 결과 차트 / 이미지 저장소) |
| `model.py` | 모델 다운로드, 전처리, 추론 함수 |
| `mnist-12.onnx` | MNIST 모델 (github.com/onnx/models, 없으면 자동 다운로드) |
| `requirements.txt` | 필요 라이브러리 |
| `Dockerfile`, `.dockerignore` | Docker 이미지 빌드 설정 |
