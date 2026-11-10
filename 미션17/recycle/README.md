# [심화] 미션17 AI 분리수거 도우미 (1팀 박단비)

채팅 화면에서 글로 물어보거나, 사진을 첨부하거나, 카메라로 찍어서 분리배출 방법을 물어볼 수 있는 Streamlit 서비스입니다.

- 사진: OpenAI CLIP 제로샷 이미지 분류 (ONNX)
- 글: 분리배출 지식 사전(47개 품목) + 다국어 MiniLM 문장 임베딩 (ONNX)
- Docker Hub: https://hub.docker.com/r/dbdbdeep02/recycle-helper

## 실행 방법

**Docker로 실행**
```bash
docker run -p 8501:8501 dbdbdeep02/recycle-helper:latest
```
브라우저에서 http://localhost:8501 접속

**로컬에서 실행** (Python 3.10)
```bash
pip install -r requirements2.txt
streamlit run app2.py
```
처음 실행할 때 ONNX 모델(약 290MB)을 Hugging Face에서 `models2/` 폴더로 자동 다운로드합니다.
(용량 때문에 제출 zip에는 모델 파일을 넣지 않았습니다.)

## 파일 구성
| 파일 | 설명 |
|---|---|
| `app2.py` | 채팅형 Streamlit 화면 (글 질문, + 버튼 사진 첨부, 카메라 촬영, 생각 과정 표시, 통계) |
| `recycle.py` | CLIP 제로샷 사진 분류, 분리배출 분류 10개 정의, 모델 다운로드 |
| `recycle_chat.py` | 글 질문 처리: 지식 사전, 규칙, 문장 임베딩 의미 검색(허브 벌점 보정), 답변 생성 |
| `requirements2.txt` | 필요 라이브러리 |
| `Dockerfile2`, `.dockerignore` | Docker 이미지 빌드 설정 (`docker build -f Dockerfile2 -t recycle-helper .`) |
