from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

Environment = Literal["development", "staging", "production"]
DEVELOPMENT: Environment = "development"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(..., alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(..., alias="OPENAI_EMBEDDING_MODEL")
    # Optional: lets the judge run on a different model from the one it grades, which
    # is what you want the day the answering model changes. Empty means "the same one".
    openai_judge_model: str = Field("", alias="OPENAI_JUDGE_MODEL")
    database_url: str = Field(..., alias="DATABASE_URL")

    app_env: Environment = Field(DEVELOPMENT, alias="APP_ENV")
    observability_enabled: bool = Field(False, alias="OBSERVABILITY_ENABLED")
    observability_sample_rate: float = Field(
        1.0, alias="OBSERVABILITY_SAMPLE_RATE", ge=0.0, le=1.0
    )
    # Captures the question and the final answer as debug events, and only in
    # development. Prompts, context, chunks and documents stay out of the telemetry
    # in every environment.
    observability_capture_content: bool = Field(
        False, alias="OBSERVABILITY_CAPTURE_CONTENT"
    )
    otel_service_name: str = Field("fcai-rag-api", alias="OTEL_SERVICE_NAME")
    otel_traces_exporter: str = Field("console", alias="OTEL_TRACES_EXPORTER")
    otel_exporter_otlp_endpoint: str = Field("", alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_exporter_otlp_headers: str = Field("", alias="OTEL_EXPORTER_OTLP_HEADERS")

    ai_allowed_models: str = Field("gpt-4.1-mini", alias="AI_ALLOWED_MODELS")
    ai_monthly_budget_usd: float = Field(10.0, alias="AI_MONTHLY_BUDGET_USD", ge=0.0)
    ai_max_output_tokens: int = Field(1200, alias="AI_MAX_OUTPUT_TOKENS", gt=0)
    ai_usage_ledger_path: Path = Field(
        Path("data/ai_usage.jsonl"), alias="AI_USAGE_LEDGER_PATH"
    )

    # Optional generic input guardrails (openai-guardrails), before the application's own
    # policies. Off by default: an experiment is not a control until someone turns it on.
    openai_guardrails_enabled: bool = Field(False, alias="OPENAI_GUARDRAILS_ENABLED")

    langfuse_public_key: str = Field("", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field("", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field("http://localhost:3000", alias="LANGFUSE_HOST")

    @property
    def judge_model(self) -> str:
        return self.openai_judge_model or self.openai_chat_model

    @property
    def psycopg_dsn(self) -> str:
        return self.database_url.replace("postgresql+psycopg://", "postgresql://")

    @property
    def allowed_models(self) -> list[str]:
        # Comma separated, not JSON: AI_ALLOWED_MODELS=gpt-4.1-mini,gpt-4.1
        return [name.strip() for name in self.ai_allowed_models.split(",") if name.strip()]

    @property
    def usage_ledger_path(self) -> Path:
        # Anchored to the project, like every other file under data/.
        path = self.ai_usage_ledger_path
        return path if path.is_absolute() else PROJECT_ROOT / path


settings = Settings()
