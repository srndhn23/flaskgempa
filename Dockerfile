# Menggunakan base image Python 3.9
FROM python:3.9

# Menetapkan working directory di dalam container
WORKDIR /app

# Menyalin file `requirements.txt` ke dalam container
COPY requirements.txt .

# Menginstall dependencies yang diperlukan
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh konten proyek Flask ke dalam container
COPY . .

# Menjalankan aplikasi Flask
CMD ["python", "main.py"]
