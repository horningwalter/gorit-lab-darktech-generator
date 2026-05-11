"""Centralized configuration loaded from environment variables and .env files."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    deepseek_api_key: SecretStr = Field(default=SecretStr(""))
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_price_input_per_mtoken_usd: float = 0.30
    deepseek_price_output_per_mtoken_usd: float = 1.20

    hf_token: SecretStr = Field(default=SecretStr(""))

    darktech_remote_url: str = Field(
        default="",
        description=(
            "If set, audio generation is delegated to a remote FastAPI worker (typically "
            "a Colab notebook with an A100). Empty string disables remote mode."
        ),
    )
    darktech_remote_api_key: SecretStr = Field(default=SecretStr(""))
    darktech_remote_timeout_s: float = 600.0

    darktech_output_dir: Path = Path("./data/output_tracks")
    darktech_cache_dir: Path = Path("./data/model_cache")
    darktech_reference_dir: Path = Path("./data/reference_tracks")
    darktech_config: str = "mvp"
    darktech_cost_hard_cap_usd: float = 0.50

    gradio_server_name: str = "0.0.0.0"
    gradio_server_port: int = 7860
    gradio_share: bool = True

    target_lufs_integrated: float = -8.0
    target_true_peak_db: float = -1.0
    output_sample_rate: int = 44100
    output_bit_depth: int = 24

    max_refinement_iterations: int = 5

    def ensure_dirs(self) -> None:
        """Create the writable directories used at runtime."""
        for p in (self.darktech_output_dir, self.darktech_cache_dir, self.darktech_reference_dir):
            p.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Lazy-loaded singleton. Tests can override via the env or by setting :data:`_settings`."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_for_tests() -> None:
    """Reset the cached settings; only intended for use in tests."""
    global _settings
    _settings = None
