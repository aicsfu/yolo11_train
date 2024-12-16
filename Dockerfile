# Используем официальный Python 3.9 образ как базовый
FROM python:3.9-slim

# Устанавливаем необходимые системные библиотеки для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем необходимые Python пакеты
RUN pip install --no-cache-dir numpy opencv-python Pillow ultralytics==8.3.38 transformers python-telegram-bot==20.0a5 torch

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем весь проект в контейнер
COPY . .

# Запускаем скрипт с ботом
CMD ["python", "bot.py"]
