"""Central config - loads .env and exposes typed settings."""

from dataclasses import dataclass
from os import getenv
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = getenv("ANTHROPIC_API_KEY", "")
    snowflake_account: str = getenv("SNOWFLAKE_ACCOUNT", "")
    snowflake_user: str = getenv("SNOWFLAKE_USER", "")
    snowflake_private_key_path: str = getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "")
    snowflake_warehouse: str = getenv("SNOWFLAKE_WAREHOUSE", "")
    snowflake_database: str = getenv("SNOWFLAKE_DATABASE", "")
    snowflake_schema: str = getenv("SNOWFLAKE_SCHEMA", "")
    snowflake_role: str = getenv("SNOWFLAKE_ROLE", "")


settings = Settings()
