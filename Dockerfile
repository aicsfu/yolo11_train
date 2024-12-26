# Используем базовый образ с поддержкой GPU
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Устанавливаем основные инструменты
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Обновляем pip
RUN python3 -m pip install --upgrade pip

# Указываем рабочую директорию
WORKDIR /workspace

# Копируем файлы из текущей директории в контейнер
COPY . /workspace

# Устанавливаем PyTorch (версия совместима с CUDA 11.8)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Устанавливаем дополнительные зависимости для работы с YOLO
RUN pip install numpy opencv-python Pillow ultralytics==8.3.38

# Устанавливаем Jupyter Lab для удобства работы
RUN pip install jupyterlab

# Команда по умолчанию (запуск Jupyter Lab на 0.0.0.0:8888)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.allow_root=True", "--ServerApp.allow_remote_access=True", "--ServerApp.token=", "--ServerApp.password="]
