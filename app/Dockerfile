FROM python:3.11.4

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download ru_core_news_lg

COPY . .

CMD ["bash", "-c", "python init_db.py && uvicorn main:app --reload --host 0.0.0.0 --port 8001"]
