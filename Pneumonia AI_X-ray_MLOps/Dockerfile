FROM python:3.10

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]