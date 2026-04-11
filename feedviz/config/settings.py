from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
BASE_DIR = Path(__file__).parent.parent
class Settings(BaseSettings):

    # LLM
    openai_api_key: str = Field(default="", env="GROQ_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")

    # File Paths
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    outputs_dir: Path = BASE_DIR / "outputs"

    feedback_csv: Path = BASE_DIR / "data" / "raw" / "feedback_train.csv"
    cleaned_csv: Path = BASE_DIR / "data" / "processed" / "feedback_cleaned.csv"

    scores_output: Path = BASE_DIR / "outputs" / "scores.json"
    rankings_output: Path = BASE_DIR / "outputs" / "rankings.json"
    insights_output: Path = BASE_DIR / "outputs" / "insights.json"
    report_output: Path = BASE_DIR / "outputs" / "reports.json"

    # Scoring Weights
    weight_rating: float = 0.5
    weight_sentiment: float = 0.3
    weight_consistency: float = 0.2

    # NLP Settings
    min_feedback_count: int = 3
    spacy_model: str = "en_core_web_sm"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings(_env_file=str(BASE_DIR / ".env"))