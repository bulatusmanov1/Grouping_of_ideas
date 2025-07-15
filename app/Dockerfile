FROM python:3.11.4

WORKDIR /app

COPY app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download ru_core_news_lg

COPY . .

RUN useradd -m app && chown -R app:app /app

USER app

# Открываем порт 8001
EXPOSE 8001

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8001"]
# docker build -t myapp -f app/Dockerfile .