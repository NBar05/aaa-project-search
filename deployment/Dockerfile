FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# CMD ["python", "/app/lib/engine.py", "&&", "python", "/app/run.py"]

CMD python /app/lib/engine.py; python /app/run.py
