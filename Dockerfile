"""
Dockerfile for the Research Assistant Network.
"""
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /app/storage/research_tasks
RUN mkdir -p /app/logs

# Expose the port
EXPOSE 5000

# Set environment variables
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=5000
ENV API_DEBUG=False

# Run the application
CMD ["python", "main.py"]
