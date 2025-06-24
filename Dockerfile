# Gunakan image Python 3.10 versi standar (non-slim)
# Ini lebih besar tetapi seringkali sudah memiliki dependensi build yang cukup
FROM python:3.10

# Set direktori kerja di dalam container
WORKDIR /app

# Pastikan apt update dijalankan dan install dependensi sistem yang dibutuhkan.
# Meskipun image non-slim, beberapa dependensi mungkin masih perlu dikonfirmasi.
# rm -rf /var/lib/apt/lists/* untuk membersihkan cache apt dan mengurangi ukuran image
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    python3-dev \
    libatlas-base-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy file requirements.txt terlebih dahulu untuk memanfaatkan Docker cache layer
# Jika requirements.txt tidak berubah, layer ini tidak akan dibangun ulang
COPY requirements.txt .

# Upgrade pip, setuptools, dan wheel ke versi terbaru
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set environment variables untuk membantu kompilasi numpy/scipy dengan BLAS/LAPACK
# Lokasi ini umum untuk pustaka di distribusi Debian/Ubuntu
ENV LDFLAGS="-L/usr/lib/x86_64-linux-gnu/openblas-base"
ENV CPPFLAGS="-I/usr/include/openblas"
ENV BLAS=/usr/lib/x86_64-linux-gnu/libopenblas.so
ENV LAPACK=/usr/lib/x86_64-linux-gnu/liblapack.so
ENV ATLAS=/usr/lib/x86_64-linux-gnu/libatlas.so

# Instal semua dependensi dari requirements.txt
# Dengan base image yang lebih lengkap dan variabel lingkungan yang disetel,
# diharapkan kompilasi berjalan lancar.
# Gunakan --no-cache-dir untuk menghindari penyimpanan cache pip di image final (mengurangi ukuran)
# --verbose dapat dipertahankan untuk debug jika masih ada masalah
RUN pip install --no-cache-dir --verbose -r requirements.txt

# Copy semua kode aplikasi ke dalam container
COPY . .


CMD sh -c "echo 🔧 Starting app on port \$PORT && uvicorn app.main:app --host 0.0.0.0 --port=${PORT:-8000}"
