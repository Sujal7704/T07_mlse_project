import os
from pathlib import Path
from typing import Tuple
from pydantic_settings import BaseSettings
from dotenv import load_dotenv  

load_dotenv("/home/ashish/Desktop/202418007/RAVSG/backend/.env")

class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    DATABASE_DIR: Path = BASE_DIR / "database"
    MEDIA_DIR: Path = BASE_DIR / "media"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # Model Configuration
    VISION_LANGUAGE_MODEL: str = "llava-hf/llava-1.5-7b-hf"
    TEXT_TO_IMAGE_MODEL: str = "runwayml/stable-diffusion-v1-5"   #"stabilityai/stable-diffusion-3.5-medium" #"stabilityai/stable-diffusion-3.5-large-turbo"     
    EMBEDDING_MODEL: str = "openai/clip-vit-base-patch16"    #"Salesforce/blip2-opt-2.7b"  # Better than CLIP for multimodal
    
    # Device Settings
    DEVICE: str = "cuda"
    USE_FP16: bool = True
    USE_4BIT: bool = True
    USE_8BIT: bool = False
    
    # Generation Parameters
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.8
    TOP_P: float = 0.9
    NUM_INFERENCE_STEPS: int = 20
    GUIDANCE_SCALE: float = 7.5
    MAX_CONTEXT_CHARS: int = 3000
    IMAGE_SIZE: int = 1024
    MAX_STORY_EXCERPT: int = 500                  # max chars for short_story
    MAX_PIXEL_COUNT: int = 1024*1024              # max total pixels
    MAX_IMAGE_DIM: Tuple[int, int] = (1024, 1024) # max width & height when resizing
    
    # RAG Configuration
    EMBEDDING_DIM: int = 512  
    TOP_K_RETRIEVAL: int = 5
    FAISS_INDEX_TYPE: str = "Flat"  # Options: "Flat", "IVFFlat"
    FAISS_NLIST: int = 100
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    JOB_TIMEOUT: int = 1200
    
    # Langfuse Configuration
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = "http://localhost:3000"
    LANGFUSE_BASE_URL: str = "http://localhost:3000"
    ENABLE_LANGFUSE: bool = False
    LANGFUSE_SAMPLE_RATE: float = 1.0  # 0.0 to 1.0
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories
        for dir_path in [self.MODELS_DIR, self.DATABASE_DIR, 
                        self.MEDIA_DIR, self.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

settings = Settings()