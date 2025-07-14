FROM python:3.12.4

WORKDIR /app

COPY app/ /app

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["uvicorn", "main_test:app", "--host", "0.0.0.0", "--port", "8000"]
