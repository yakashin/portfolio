import os
import requests
from PIL import Image
from io import BytesIO
import torch
from PIL import ImageFile
from torchvision import models
from torchvision.transforms import v2
from datetime import datetime
import torch.nn as nn

from config import ModelConfig


class ModelRotationPredictor:
    """
    Класс для предсказания угла поворота картинки.
    Класс основан на дообученной на фотографиях справок доноров, в основе модели лежит модель EfficientNet.

    def __init__: инициализация модели, подготовка изображения в удобный формат tensor.
    def load_image: загрузка изображения из сети или из локального хранилища
    def rotate_and_save_image: функция поворота и сохранения изображения
    def predict_rotation: функция предсказания поворота изображения
    def process_image: функция для процесса предсказания поворота с выводом результата
    """
    def __init__(self, model_path: str, device: str=ModelConfig().device):
        """
        Инициализация модели EfficientNet для классификации поворота изображений и предобработка данных.

        :param model_path: Путь к файлу с обученными весами модели (формат .pth)
        :param device: Устройство для выполнения вычислений

        Описание:
        - Загружает модель EfficientNet с предварительно обученными весами и дообучением на исходных данных.
        - Заменяет последний полносвязный слой модели для предсказания четырёх классов (0, 90, 180, 270 градусов).
        - Устанавливает модель в режим оценки (evaluation mode) для предотвращения обучения.

        Предобработка изображения включает:
        - Обрезка изображения по центру до 224x224 пикселей.
        - Преобразование изображения в тензор.
        - Нормализацию значений тензора с использованием стандартных средних и стандартных отклонений.
        """
        self.model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 4)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.model.to(device)
        self.model.eval()  # Переводим модель в режим оценки

        # Предобработка изображения
        self.preprocess = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    @staticmethod
    def load_image(image_path: str, type_input: str=ModelConfig().default_type_input) ->  ImageFile.ImageFile|str:
        """
        Загружает изображение из локального пути или по URL и возвращает объект изображения (PIL.Image).

        :param image_path: Путь к изображению (локальный путь или URL)
        :param type_input: Тип входных данных: 'local' для локального пути или 'url' для ссылки (по умолчанию 'local')

        :return: Объект изображения (PIL.Image) в случае успеха или строка с сообщением об ошибке в случае сбоя.

        Пример использования:
        1. Для загрузки изображения из URL:
            image = load_image("https://example.com/image.jpg", type_input='url')

        2. Для загрузки локального изображения:
            image = load_image("path/to/local/image.jpg")

        Ошибки:
        - ValueError: если URL не удаётся загрузить (неправильный ответ сервера).
        - FileNotFoundError: если файл по указанному локальному пути не найден.
        - Любая другая ошибка возвращается как строка с описанием проблемы.
        """
        try:
            if type_input=='url':
                response = requests.get(image_path)
                if response.status_code == 200:
                    # Если ответ от сервера успешен, загружаем изображение
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    return img
                else:
                    raise ValueError(
                        f"Не удалось загрузить изображение по ссылке: {image_path}. Код ответа: {response.status_code}")
            elif os.path.exists(image_path):
                # Если это локальный путь, проверяем существование файла
                img = Image.open(image_path).convert('RGB')
                return img
            else:
            # Если файл не найден по пути
                raise FileNotFoundError(f"Файл по пути '{image_path}' не найден.")

        except Exception as e:
        # В случае любой ошибки возвращаем сообщение об ошибке
            return f"Ошибка при загрузке изображения: {e}"

    @staticmethod
    def rotate_and_save_image(image: ImageFile.ImageFile, angle: int, save_dir: str=ModelConfig().save_dir) -> str:
        """
        Поворачивает изображение на указанный угол, сохраняет его в указанную папку с именем на основе текущей даты и времени.

        :param image: объект PIL.Image (исходное изображение)
        :param angle: угол поворота изображения (0, 90, 180, 270 градусов)
        :param save_dir: путь к папке, в которую нужно сохранить изображение (по умолчанию './src/results/')

        :return: полный путь к сохраненному файлу

        Описание:
        - Поворачивает изображение на заданный угол с использованием метода rotate() из PIL.
        - Проверяет, существует ли директория для сохранения; если нет, создает её.
        - Генерирует уникальное имя файла на основе текущей даты и времени в формате 'image_YYYYMMDD_HHMMSS.png'.
        - Сохраняет повернутое изображение в указанную директорию в формате PNG.
        - Возвращает полный путь к сохраненному файлу.

        Пример использования:
        rotated_image_path = rotate_and_save_image(image, 90)
        """
        # Поворачиваем изображение на указанный угол
        rotated_image = image.rotate(-angle, expand=True)

        # Убедимся, что директория для сохранения существует
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Генерация имени файла на основе текущей даты и времени

        current_time = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        file_name = f"image_{current_time}.png"

        # Полный путь к файлу
        file_path = os.path.join(save_dir, file_name)

        # Сохраняем изображение
        rotated_image.save(file_path)

        return file_path

    def predict_rotation(self, img: ImageFile.ImageFile, device: str=ModelConfig().device) -> int:
        """
        Предсказание угла поворота изображения (0, 90, 180, 270 градусов) с помощью обученной модели.

        :param img: объект PIL.Image, представляющий изображение, для которого нужно предсказать угол поворота.
        :param device: Устройство для выполнения вычислений

        :return: предсказанный угол поворота (одно из значений: 0, 90, 180, 270 градусов).

        Описание:
        - Преобразует изображение в тензор с помощью метода предобработки (self.preprocess), затем добавляет измерение батча.
        - Передает изображение через модель, находящуюся в режиме оценки (без обновления градиентов).
        - Определяет класс, соответствующий углу поворота, с помощью argmax по оси предсказаний.
        - Сопоставляет предсказанный класс с соответствующим углом поворота:
          - 0 для класса 0,
          - 90 для класса 1,
          - 180 для класса 2,
          - 270 для класса 3.

        Пример использования:
        predicted_angle = model.predict_rotation(img)
        """
        # Преобразуем изображение в тензор
        img_tensor = self.preprocess(img).unsqueeze(0).to(device)  # Добавляем батч
        with torch.no_grad():
            output = self.model(img_tensor)
        predicted_class = output.argmax(dim=1).item()

        # Углы, соответствующие классам
        angles = {0: 0, 1: 90, 2: 180, 3: 270}
        return angles[predicted_class]

    def process_image(self, image_path: str, type_input: str=ModelConfig().default_type_input) -> tuple:
        """
        Обрабатывает изображение: загружает его, предсказывает угол поворота, поворачивает и сохраняет результат.

        :param image_path: Путь к изображению (может быть URL или локальный путь).
        :param type_input: Тип входных данных: 'local' для локального файла или 'url' для изображения по ссылке (по умолчанию 'local').

        :return: Кортеж, содержащий:
            - Предсказанный угол поворота (одно из значений: 0, 90, 180, 270 градусов).
            - Полный путь к сохраненному повернутому изображению.

        Описание:
        1. Загружает изображение с указанного пути. Если тип входных данных 'url', загружает изображение по ссылке, если 'local' — с локального пути.
        2. Использует модель для предсказания угла поворота изображения (0, 90, 180, 270 градусов).
        3. Поворачивает изображение на предсказанный угол.
        4. Сохраняет повернутое изображение с уникальным именем на основе текущей даты и времени.
        5. Возвращает предсказанный угол поворота и путь к сохраненному изображению.

        Пример использования:
        predicted_angle, saved_image_path = model.process_image("path/to/image.jpg", type_input='local')
        """
        # 1. Загружаем изображение
        img = self.load_image(image_path=image_path, type_input=type_input)

        # 2. Предсказываем угол поворота
        predicted_angle = self.predict_rotation(img, ModelConfig().device)

        # 3. Поворачиваем изображение и сохраняем
        name_files = self.rotate_and_save_image(image=img, angle=predicted_angle)

        return predicted_angle, name_files
