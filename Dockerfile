# Gunakan base image Python dengan TensorFlow support
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies untuk TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy kode aplikasi
COPY Main.py .
COPY .env* ./

# Copy model dan data (opsional, bisa di-mount via volume)
# COPY best_model.keras ./
# COPY model_cek_gula.keras ./
# COPY nutrisi_jajanan.csv ./

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Default command
CMD ["uvicorn", "Main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]