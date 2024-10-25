# Dockerfile
FROM python:3.12-slim

# Set up the working directory
WORKDIR /app

# Install the required packages
RUN pip install --upgrade torch transformers huggingface_hub accelerate

# Copy the application code
COPY ./app /app

CMD ["python", "falcon_inference.py"]
