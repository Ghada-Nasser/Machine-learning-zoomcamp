FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir .

COPY app.py predict.py ./
COPY models/ ./models/

EXPOSE 5000

CMD ["python", "app.py"]