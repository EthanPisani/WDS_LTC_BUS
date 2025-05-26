FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir streamlit requests

COPY frontend.py .
# export api url as an environment variable
ENV API_URL=http://backend:5240
EXPOSE 5241

CMD ["streamlit", "run", "frontend.py", "--server.port=5241", "--server.address=0.0.0.0"]
