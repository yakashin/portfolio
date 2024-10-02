from pydantic import BaseModel


class ModelConfig(BaseModel):
    """
    Configuration for EfficientNet
    """
    models_path: str = "./resources/rotation_efnet_classifier.pth"
    save_dir: str = './src/results/'
    default_type_input: str = 'url'
    device: str = 'cpu'


class AppConfig(BaseModel):
    """
    Configuration for FastAPI App
    """
    host: str = '0.0.0.0'
    port: int = 8000
