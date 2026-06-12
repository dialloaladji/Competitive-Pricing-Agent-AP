from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    environment: str = "development"
    mock_mode: bool = False
    debug: bool = True

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
    redis_url: str = "redis://localhost:6379/0"

    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    llm_provider: str = "llamacpp"
    llama_cpp_base_url: str = "https://api.groq.com/openai/v1"
    llama_cpp_model: str = "llama-3.1-8b-instant"
    llama_cpp_api_key: str = ""
    llm_max_tokens: int = 1500

    tavily_api_key: str = ""
    serpapi_api_key: str = ""
    web_search_primary: str = "tavily"
    web_search_fallback: str = "serpapi"
    web_search_max_results: int = 3

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    price_track_interval_minutes: int = 60
    alert_check_interval_minutes: int = 15

    rate_limit_per_minute: int = 60
    cache_ttl_seconds: int = 300

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
