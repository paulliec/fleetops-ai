"""Snowflake connection using key-pair auth."""

from pathlib import Path

import snowflake.connector
from cryptography.hazmat.primitives import serialization

from config.settings import settings


def _load_private_key():
    """Read PEM private key file, return DER bytes for the connector."""
    key_path = Path(settings.snowflake_private_key_path).expanduser()
    pem_data = key_path.read_bytes()
    private_key = serialization.load_pem_private_key(pem_data, password=None)
    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def get_connection():
    params = dict(
        account=settings.snowflake_account,
        user=settings.snowflake_user,
        private_key=_load_private_key(),
        warehouse=settings.snowflake_warehouse,
        database=settings.snowflake_database,
        schema=settings.snowflake_schema,
    )
    if settings.snowflake_role:
        params["role"] = settings.snowflake_role
    return snowflake.connector.connect(**params)


def test_connection():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_WAREHOUSE(), CURRENT_DATABASE()")
        user, role, warehouse, database = cur.fetchone()
        print(f"user:      {user}")
        print(f"role:      {role}")
        print(f"warehouse: {warehouse}")
        print(f"database:  {database}")
    finally:
        conn.close()


if __name__ == "__main__":
    test_connection()
