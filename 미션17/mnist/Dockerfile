# 손글씨 숫자 인식 Streamlit 서비스 Docker 이미지
# 빌드: docker build -t <도커허브아이디>/mnist-streamlit:latest .
# 실행: docker run -p 8501:8501 <도커허브아이디>/mnist-streamlit:latest

# 로컬 개발 환경과 같은 Python 3.10 (slim: 용량이 작은 버전)
FROM python:3.10-slim

# 파이썬 출력 버퍼링 끄기, .pyc 파일 만들지 않기
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 라이브러리를 먼저 설치합니다. 코드만 바뀌었을 때는 이 단계가 캐시되어 빌드가 빨라집니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY model.py app.py ./

# 빌드할 때 MNIST ONNX 모델을 미리 내려받아 이미지에 포함시킵니다.
# (컨테이너가 시작될 때마다 다시 내려받지 않아도 됨)
RUN python -c "from model import download_model; download_model()"

# Streamlit 기본 포트
EXPOSE 8501

# 컨테이너 상태 확인 (Streamlit 헬스체크 엔드포인트)
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# 외부에서 접속할 수 있도록 0.0.0.0으로 실행
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
