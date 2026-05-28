from PIL import Image
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Получаем токен из переменной окружения
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("Токен HF_TOKEN не найден в файле .env")

# Создание клиента для взаимодействия с моделью
client = InferenceClient(token=HF_TOKEN)

# Генерация изображения по текстовому описанию
image = client.text_to_image(
    model="black-forest-labs/FLUX.1-schnell",
    prompt="A futuristic cityscape under a purple sky, glowing neon lights, flying cars, highly detailed digital art"
)

# Сохранение изображения
image.save("generated_image.png")

# Вывод сообщения об успешном сохранении
print("Изображение успешно сохранено как 'generated_image.png'")