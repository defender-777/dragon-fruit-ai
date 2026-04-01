# Base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (for PIL / torch)
RUN apt-get update --fix-missing && apt-get install -y \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --default-timeout=1000 --retries=5 \
    -f https://download.pytorch.org/whl/torch_stable.html \
    -r requirements.txt

# Copy all backend code
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]