FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir flask pandas scikit-learn matplotlib boto3 awscli
EXPOSE 5000

CMD ["python", "app.py"]
