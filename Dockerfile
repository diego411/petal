# syntax=docker/dockerfile:1
FROM python:3.10-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
#CMD [ "python3", "-m" , "flask", "--app", "src/main", "run", "--host=0.0.0.0"]
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "src.main:create_app()"]