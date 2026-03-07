FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
COPY citadel_gateway/ citadel_gateway/

RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["citadel-gateway"]
