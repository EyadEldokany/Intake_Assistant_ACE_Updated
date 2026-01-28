# ACE Medical Assistant - Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY playbook.py .
COPY curator_agent.py .
COPY reflector_agent.py .
COPY summarizer_agent.py .
COPY playbook_data.json .

# Create logs directory
RUN mkdir -p logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_BASE_URL=https://api.ollama.cloud

# Run the application
CMD ["python", "app.py"]