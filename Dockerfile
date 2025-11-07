# 1. 사용할 파이썬 이미지 설정
FROM python:3.11-slim

# 2. 작업 디렉토리를 /app으로 설정
WORKDIR /app

# 3. requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 나머지 모든 파일 복사
COPY . .

# 5. 서버 포트 노출
EXPOSE 8000

# 6. 서버 실행 명령어 지정 (파일명: mcp_server.py, 인스턴스명: app 가정)
# [핵심]: 파일 이름(mcp_server)과 인스턴스 이름(app)을 맞춰주세요.
CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]