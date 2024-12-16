import logging
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from io import BytesIO
from telegram import Update
from telegram.ext import (
    Application,  # Новый способ создания приложения
    CommandHandler,
    MessageHandler,
    filters,  # Используем filters с маленькой буквы
    CallbackContext,
)

# Установим базовую конфигурацию логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загружаем модели
model_yolo = YOLO(r"best.pt", verbose=False)  # Путь к модели YOLO
processor_trocr = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model_trocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")


async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "Привет! Отправьте мне фото с номерным знаком для распознавания."
    )


async def process_image(update: Update, context: CallbackContext):
    # Проверяем наличие фото в сообщении
    if update.message.photo:
        # Получаем фото с максимальным разрешением
        photo = update.message.photo[-1]
        # Асинхронно получаем файл
        photo_file = await photo.get_file()

        # Скачиваем файл как байты
        photo_bytes = await photo_file.download_as_bytearray()

        # Преобразуем байты в изображение через PIL
        try:
            pil_image = Image.open(BytesIO(photo_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"Ошибка при загрузке изображения: {e}")
            await update.message.reply_text("Не удалось загрузить изображение.")
            return

        # Преобразуем PIL изображение в OpenCV формат
        img = np.array(pil_image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Выполняем предсказание (детекция номерного знака)
        results_list = model_yolo.predict(
            img, conf=0.21, verbose=False, device="cpu", workers=1
        )

        # Проходим по результатам детекции
        for result in results_list:
            boxes = result.boxes.xywh  # Координаты боксов
            confidences = result.boxes.conf  # Доверие для каждого бокса
            labels = result.boxes.cls  # Индексы классов
            class_names = result.names

            for i, box in enumerate(boxes):
                if (
                    class_names[int(labels[i])] == "class0"
                ):  # Если класс это номерной знак
                    x, y, w, h = box
                    x1, y1, x2, y2 = (
                        int(x - w / 2),
                        int(y - h / 2),
                        int(x + w / 2),
                        int(y + h / 2),
                    )

                    # Вырезаем номерной знак из исходного изображения
                    plate_img = img[y1:y2, x1:x2]

                    # Проверка на пустое изображение
                    if plate_img.size == 0:
                        continue

                    # Конвертируем в формат RGB для TrOCR
                    plate_pil = Image.fromarray(
                        cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                    )

                    # Обрабатываем изображение с помощью TrOCR
                    pixel_values = processor_trocr(
                        plate_pil, return_tensors="pt"
                    ).pixel_values
                    generated_ids = model_trocr.generate(pixel_values)
                    generated_text = processor_trocr.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0]

                    # Рисуем прямоугольник вокруг номерного знака
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Добавляем текст поверх бокса
                    font_scale = 1.5
                    font_thickness = 3
                    text_size = cv2.getTextSize(
                        generated_text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        font_thickness,
                    )[0]
                    text_w, text_h = text_size

                    # Белый фон для текста
                    cv2.rectangle(
                        img,
                        (x1, y1 - text_h - 10),
                        (x1 + text_w, y1),
                        (255, 255, 255),
                        -1,
                    )
                    # Подпись с распознанным текстом
                    cv2.putText(
                        img,
                        generated_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        font_thickness,
                    )

        # Преобразуем обработанное изображение в формат для отправки
        _, buffer = cv2.imencode(".jpg", img)
        img_bytes = buffer.tobytes()

        # Отправляем обработанное изображение обратно пользователю
        await update.message.reply_photo(
            photo=BytesIO(img_bytes), caption="Распознанный номерной знак"
        )

    else:
        await update.message.reply_text(
            "Пожалуйста, отправьте изображение с номерным знаком."
        )


def main():
    # Вставьте сюда ваш токен
    TOKEN = "7287622548:AAGBEwjd5nhQS-XhGv4sa6Ihc06LOfZlHM4"

    # Создаем объект Application
    application = Application.builder().token(TOKEN).build()

    # Обработчики команд и сообщений
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, process_image))

    # Запускаем бота
    application.run_polling()


if __name__ == "__main__":
    main()
