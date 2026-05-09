"""
Retrieves chat history from PostgreSQL for session context.

Connects to the chat_sessions table and returns the last N Q&A pairs
from the chat_history JSONB column for a given session_id.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


class SessionContext:
    """Fetches the last N Q&A pairs from chat_sessions.chat_history."""

    def __init__(self, config=None):
        self._config = config

    def _conn_params(self) -> dict:
        if self._config and hasattr(self._config, "postgres"):
            p = self._config.postgres
            return {
                "host": p.host,
                "port": p.port,
                "dbname": p.db,
                "user": p.user,
                "password": p.password,
                "sslmode": p.sslmode,
                "connect_timeout": 5,
            }
        return {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "dbname": os.getenv("POSTGRES_DB", "chatdb"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", ""),
            "sslmode": os.getenv("POSTGRES_SSL", "prefer"),
            "connect_timeout": 5,
        }

    def get_last_n(self, session_id: str, n: int = 3) -> list[dict]:
        """
        Return the last n Q&A entries from chat_sessions.chat_history.

        Each entry is a dict with at least 'question' and 'answer' keys.
        Returns an empty list when session_id is blank or on any DB error
        so the caller always degrades gracefully to no-context mode.
        """
        if not session_id or not session_id.strip():
            return []

        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            with psycopg2.connect(**self._conn_params()) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT chat_history FROM chat_sessions WHERE id = %s",
                        (session_id,),
                    )
                    row = cur.fetchone()

            if not row or not row["chat_history"]:
                return []

            history = row["chat_history"]
            if isinstance(history, str):
                history = json.loads(history)

            # Keep only the last n entries
            return history[-n:] if len(history) > n else history

        except Exception as exc:
            logger.warning(
                "Failed to fetch chat history for session %s: %s", session_id, exc
            )
            return []
