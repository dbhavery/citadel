FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
COPY citadel_trace/ citadel_trace/

RUN pip install --no-cache-dir ".[server]"

EXPOSE 8081

CMD ["citadel-trace", "serve"]
