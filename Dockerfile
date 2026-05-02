# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
ENV PORT 8501

# Set up the working directory
WORKDIR /app

# Install system dependencies (e.g. for building C extensions or running ChromaDB safely)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./

# Install pip requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy local code to the container image.
COPY . ./

# Expose port for Streamlit
EXPOSE 8501

# Run the web service on container startup.
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
