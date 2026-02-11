FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Railway akan memberikan port secara otomatis melalui environment variable
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]