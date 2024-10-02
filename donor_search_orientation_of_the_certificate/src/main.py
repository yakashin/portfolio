import warnings
from fastapi import FastAPI, File, UploadFile, Request, Form
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os
from pathlib import Path

from config import ModelConfig, AppConfig
from model.model import ModelRotationPredictor

warnings.filterwarnings('ignore')

app = FastAPI()
templates = Jinja2Templates(directory="src/templates")

# Указание пути для статических файлов
app.mount("/src", StaticFiles(directory="src"), name="static")


@app.get("/testing", response_class=HTMLResponse)
def load_form(request: Request):
    """
    Загружает HTML-шаблон формы для загрузки файла.

    :param request: Объект запроса FastAPI, содержащий информацию о текущем запросе.
    :return: Шаблон HTML-страницы (upload.html) с переданным объектом запроса для отображения формы загрузки.

    Описание:
    Функция возвращает страницу с формой для загрузки файла, используя шаблон "upload.html".
    В качестве контекста передаётся объект request, необходимый для рендеринга шаблона.
    """
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/")
def send_info() -> dict:
    """
    Endpoint который возвращает информацию о сервисе
    :return: dict - информация о сервисе
    """
    return {'message': 'Сервис по определению положения справки на фотографии и повороту, при необходимости '
                       'Для выполнения предсказания используйте endpoint /predict '
                       'Для визуального тестирования перейдите в /testing '
                       'Для получения информации о микросервисе перейдите в /docs'}


@app.post("/predict")
def predict(image_paths: str, type_input: str) -> dict:
    """
    Endpoint предсказывает угол поворота изображения, поворачивает его и возвращает результат.

    :param image_paths: Путь к изображению (может быть URL или локальный путь).
    :param type_input: Тип входных данных: 'local' для локального файла или 'url' для изображения по ссылке.

    :return: Словарь с информацией о результате обработки, содержащий:
        - 'message': Статус операции (например, "Success flipping").
        - 'predictions': Предсказанный угол поворота изображения (0, 90, 180, 270 градусов).
        - 'files_path': Путь к файлу с повернутым изображением.

    Описание:
    1. Инициализирует класс ImageRotationPredictor с заранее обученной моделью для предсказания угла поворота изображения.
    2. Вызывает метод process_image для предсказания угла поворота и сохранения повернутого изображения.
    3. Возвращает словарь с успешным сообщением, предсказанным углом поворота и путём к повернутому изображению.

    Пример использования:
    result = predict("path/to/image.jpg", "local")
    """

    rotation_predictor = ModelRotationPredictor(ModelConfig().models_path, ModelConfig().device)
    rotated_image = rotation_predictor.process_image(image_paths, type_input)
    return {'message': 'Success flipping', 'predictions angle': rotated_image[0], 'files_path': rotated_image[1]}


@app.post("/uploadfile/")
async def handle_upload(request: Request, file: UploadFile = File(...)):
    """
    Обрабатывает загрузку файла, предсказывает угол поворота изображения и отображает результат.

    :param request: Объект запроса FastAPI, содержащий информацию о текущем запросе.
    :param file: Загруженный файл, который пользователь отправляет через форму.

    :return: Шаблон HTML-страницы с результатом, содержащий:
        - предсказанный угол поворота,
        - ссылку на повернутое изображение.

    Описание:
    1. Проверяет наличие директории для сохранения файла и создает её, если не существует.
    2. Сохраняет загруженный файл в локальной директории.
    3. Вызывает функцию `predict` для предсказания угла поворота изображения.
    4. Возвращает HTML-шаблон с изображением и предсказанным углом поворота.
    """

    # Убедимся, что директория для сохранения существует
    if not os.path.exists('./src/static'):
        os.makedirs('./src/static')

    # Сохранение загруженного файла
    file_path = f"./src/static/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Вызов функции predict для предсказания угла поворота
    result = predict(file_path, type_input="local")

    # Путь к изображению и предсказанный угол
    rotated_image_path = f"./results/{Path(result['files_path']).name}"
    predicted_angle = result['predictions angle']

    return templates.TemplateResponse("result.html", {
        "request": request,
        "angle": predicted_angle,
        "image_url": rotated_image_path
    })

if __name__ == "__main__":
    uvicorn.run(app, host=AppConfig().host, port=AppConfig().port)
