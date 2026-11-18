FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app
COPY ./models ./models

RUN mkdir -p /app/static /tmp/clipper /app/models && \
    if [ ! -f /app/models/face_detection_yunet_2023mar.onnx ]; then \
        curl -fsSL https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
        -o /app/models/face_detection_yunet_2023mar.onnx; \
    fi && \
    echo "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  /app/models/face_detection_yunet_2023mar.onnx" | sha256sum -c -
