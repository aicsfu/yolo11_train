docker build -t yolo11-container .

docker run --gpus all -it --rm -p 8888:8888 yolo11-container