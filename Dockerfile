FROM python:3.12-slim

# System deps for TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Hugging Face Spaces uses port 7860
ENV PORT=7860
EXPOSE 7860

# Start Flask
CMD ["python", "backend/app.py"]
