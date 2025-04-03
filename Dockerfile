# Dockerfile
FROM python:3.11-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy source code, requirements and data
COPY src/ src/
COPY requirements requirements/
COPY data/ data/
COPY tests tests/
COPY pytest.ini pytest.ini

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements/requirements.txt
RUN pip install pytest
