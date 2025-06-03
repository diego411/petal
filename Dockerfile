# syntax=docker/dockerfile:1
FROM python:3.10-slim-bullseye

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libpq-dev \
        python3-dev \
        gcc \
        tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "-m", "flask", "--app", "src/main", "run", "--host=0.0.0.0"]
