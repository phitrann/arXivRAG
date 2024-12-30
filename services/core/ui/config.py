import dotenv

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

dotenv.load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra="ignore")
    # RAG Core
    RAG_CORE_URL: str = Field(default="http://localhost:8800")

    # LLM Settings
    LLM_SERVING_URL: str = Field(default="http://localhost:8001")
    RERANK_SERVING_URL: str = Field(default="http://localhost:8002")
    EMB_SERVING_URL: str = Field(default="http://localhost:8003")
    MAX_NEW_TOKENS: int = Field(default=512)

    # Milvus
    MILVUS_URL: str = Field(default="http://localhost:19530")

    # MinIO
    MINIO_URL: str = Field(default="localhost:9800")
    MINIO_ACCESS_KEY: str = Field(default="minioadmin")
    MINIO_SECRET_KEY: str = Field(default="minioadmin")

    # Retrieve
    RETRIEVER_URL: str = Field(default="http://localhost:8080")
    SIMILARITY_TOP_K: int = Field(default=8)
    FUSION_TOP_K: int = Field(default=5)
    RERANK_TOP_K: int = Field(default=3)
    NUM_QUERY: int = Field(default=1)

    VECTOR_STORE_COLLECTION: str = Field(default="arxiv_papers")

    # Embedding
    CHUNK_SIZE: int = Field(default=256)
    EMBEDDING_DIM: int = Field(default=768) 

settings = Settings()