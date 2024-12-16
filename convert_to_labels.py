import os
import json
from PIL import Image


# Функция для преобразования координат в формат YOLO
def convert_to_yolo(box, img_width, img_height):
    x_coords = [point[0] for point in box]
    y_coords = [point[1] for point in box]

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


# Директория для сохранения меток
output_dir = r"C:\vkcv2022-contest-02-carplates\data\labels"
os.makedirs(output_dir, exist_ok=True)

# Директория с изображениями
images_dir = r"C:\vkcv2022-contest-02-carplates\data\train"

# Чтение исходного JSON
input_json_path = r"C:\vkcv2022-contest-02-carplates\data\train.json"
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Преобразование данных
for item in data:
    image_path = os.path.join(images_dir, os.path.basename(item["file"]))

    # Обработка исключений при открытии изображений
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        continue  # Переходим к следующему изображению, если ошибка

    file_name = os.path.basename(item["file"]).split(".")[0] + ".txt"
    output_path = os.path.join(output_dir, file_name)

    with open(output_path, "w") as txt_file:
        for num in item["nums"]:
            box = num["box"]
            yolo_format = convert_to_yolo(box, img_width, img_height)
            txt_file.write(yolo_format + "\n")

# Уведомление о завершении
print(f"Labels successfully created in '{output_dir}' directory.")
