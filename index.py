# index.py
# Веб-приложение Flask для генерации изображений через Hugging Face

import os
from flask import Flask, request, render_template, Response, jsonify
from dotenv import load_dotenv
from huggingface_hub import InferenceClient, InferenceTimeoutError
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
import io

# Загружаем переменные окружения из .env
load_dotenv()

# Инициализируем Flask
app = Flask(__name__)

# Константы
MIN_SIZE = 256
MAX_SIZE = 1024
MULTIPLE = 32

# Получаем токен из .env и создаём клиента Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN не найден в .env файле")

client = InferenceClient(token=HF_TOKEN)

# Маршрут /login: возвращает JSON с логином автора
@app.route("/login", methods=["GET"])
def login():
    return jsonify({"author": "1160468"})

# Маршрут /makeimage: обрабатывает GET и POST
@app.route("/makeimage", methods=["GET", "POST"])
def makeimage():
    if request.method == "GET":
        return render_template(
            "makeimage.html",
            width=512,
            height=512,
            text="cyberpunk cat with neon glasses",
            message=""
        )

    # POST: обработка формы
    width_str = request.form.get("width", "").strip()
    height_str = request.form.get("height", "").strip()
    text = request.form.get("text", "").strip()

    # Валидация ширины и высоты
    try:
        width = int(width_str)
        height = int(height_str)
    except ValueError:
        message = "Invalid image size"
        return render_template(
            "makeimage.html",
            width=width_str,
            height=height_str,
            text=text,
            message=message
        )

    if width < MIN_SIZE or height < MIN_SIZE:
        message = "Invalid image size"
        return render_template(
            "makeimage.html",
            width=width,
            height=height,
            text=text,
            message=message
        )

    if width > MAX_SIZE or height > MAX_SIZE:
        message = "Invalid image size"
        return render_template(
            "makeimage.html",
            width=width,
            height=height,
            text=text,
            message=message
        )

    if width % MULTIPLE != 0 or height % MULTIPLE != 0:
        message = "Width and height must be multiples of 32"
        return render_template(
            "makeimage.html",
            width=width,
            height=height,
            text=text,
            message=message
        )

    if not text:
        message = "Prompt text is required."
        return render_template(
            "makeimage.html",
            width=width,
            height=height,
            text=text,
            message=message
        )

    try:
        # Генерация изображения с таймаутом
        image = client.text_to_image(
            prompt=text,
            model="black-forest-labs/FLUX.1-schnell",
            width=width,
            height=height,
            timeout=30  # секунд
        )

        # Конвертация в RGB при необходимости
        if image.mode in ("RGBA", "LA"):
            image = image.convert("RGB")

        # Сохранение в буфер
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        buf.seek(0)

        return Response(
            buf.getvalue(),
            mimetype="image/jpeg"
        )

    except InferenceTimeoutError:
        message = "Model generation failed: Generation timed out."
        return render_template(
            "makeimage.html",
            width=width,
            height=height,
            text=text,
            message=message
        )
    except HfHubHTTPError as e:
        if e.response.status_code == 429:
            message = "Model generation failed: Rate limit exceeded."
        else:
            message = f"Model generation failed: {e.message}"
        return render_template(
            "makeimage.html",
            width=width,
            height=height,
            text=text,
            message=message
        )
    except Exception as e:
        message = f"Model generation failed: {str(e)}"
        return render_template(
            "makeimage.html",
            width=width,
            height=height,
            text=text,
            message=message
        )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)