FROM python:3.11-slim

WORKDIR /app

# Instala dependencias de sistema
RUN apt-get update && apt-get install -y gcc default-libmysqlclient-dev sqlite3

# Copia requerimientos y los instala
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia el código fuente (incluye .files y externals.json)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
