"""
memory/memory_manager.py

Async MongoDB session store using Motor.
Handles full session lifecycle: create, load, update, expire, TTL index.

Session document schema:
{
    "session_id": str,
    "summaries": [],
    "last_messages": {},   # {"user": str, "assistant": str}
    "message_count": int,
    "created_at": datetime,
    "last_active": datetime,
    "is_expired": bool
}
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import motor.motor_asyncio

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class MemoryManager:
    def __init__(self, uri: str = None, db_name: str = None):
        self.uri = uri or settings.MONGODB_URI
        self.db_name = db_name or settings.MONGODB_DB_NAME
        self.is_configured = bool(self.uri and self.uri.strip())

        if not self.is_configured:
            logger.warning(
                "No MONGODB_URI configured — memory will not be persisted. "
                "Set MONGODB_URI in .env to enable session memory."
            )
            self._client = None
            self._db = None
        else:
            self._client = motor.motor_asyncio.AsyncIOMotorClient(self.uri)
            self._db = self._client[self.db_name]

    # ─── Initialization ────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Create indexes on startup.
        TTL index on last_active — MongoDB auto-removes documents after
        SESSION_EXPIRY_HOURS hours of inactivity.
        """
        if not self.is_configured:
            return
        try:
            await self._db.sessions.create_index(
                "last_active",
                expireAfterSeconds=settings.SESSION_EXPIRY_HOURS * 3600,
                name="session_ttl",
            )
            logger.info(
                f"TTL index created on sessions.last_active "
                f"(expires after {settings.SESSION_EXPIRY_HOURS}h)"
            )
        except Exception as e:
            logger.error(f"Failed to create TTL index: {e}")

    async def close(self) -> None:
        """Close the Motor client gracefully."""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")

    # ─── Session CRUD ──────────────────────────────────────────────────────────

    async def create_session(self, session_id: str) -> dict:
        """
        Insert a fresh session document into MongoDB.
        Every new WebSocket connection must call this with a unique session_id.
        """
        if not self.is_configured:
            return _empty_session(session_id)
        try:
            now = datetime.now(timezone.utc)
            doc = {
                "session_id": session_id,
                "summaries": [],
                "last_messages": {},
                "message_count": 0,
                "created_at": now,
                "last_active": now,
                "is_expired": False,
            }
            await self._db.sessions.insert_one(doc)
            logger.info(f"[{session_id}] Session created in MongoDB")
            return doc
        except Exception as e:
            logger.error(f"[{session_id}] Failed to create session: {e}")
            return _empty_session(session_id)

    async def get_session(self, session_id: str) -> Optional[dict]:
        """
        Load the session document.
        Returns None if not found or expired (application-level check).
        MongoDB TTL will auto-remove truly expired docs, but we also
        check manually to handle the window between TTL runs.
        """
        if not self.is_configured:
            return None
        try:
            doc = await self._db.sessions.find_one({"session_id": session_id})
            if not doc:
                return None

            last_active = doc.get("last_active")
            if last_active:
                if last_active.tzinfo is None:
                    last_active = last_active.replace(tzinfo=timezone.utc)
                expiry = last_active + timedelta(hours=settings.SESSION_EXPIRY_HOURS)
                if datetime.now(timezone.utc) > expiry:
                    logger.info(f"[{session_id}] Session expired — deleting")
                    await self._delete_session(session_id)
                    return None

            return doc
        except Exception as e:
            logger.error(f"[{session_id}] Failed to get session: {e}")
            return None

    async def _delete_session(self, session_id: str) -> None:
        """Hard-delete an expired session document."""
        try:
            await self._db.sessions.delete_one({"session_id": session_id})
            logger.info(f"[{session_id}] Expired session deleted")
        except Exception as e:
            logger.error(f"[{session_id}] Failed to delete session: {e}")

    async def touch_session(self, session_id: str) -> None:
        """Update last_active timestamp on every turn."""
        if not self.is_configured:
            return
        try:
            await self._db.sessions.update_one(
                {"session_id": session_id},
                {"$set": {"last_active": datetime.now(timezone.utc)}},
            )
        except Exception as e:
            logger.error(f"[{session_id}] Failed to touch session: {e}")

    # ─── Message Count ─────────────────────────────────────────────────────────

    async def increment_message_count(self, session_id: str) -> int:
        """
        Atomically increment message_count and return the value BEFORE increment.
        The pre-increment count is what drives phase logic:
          0 → Phase 1 (first message, no history)
          1 → Phase 2 (second message, include last exchange)
          2+ → Phase 3 (summaries only)
        """
        if not self.is_configured:
            return 0
        try:
            await self.touch_session(session_id)
            # find_one_and_update returns the document state BEFORE the update by default
            result = await self._db.sessions.find_one_and_update(
                {"session_id": session_id},
                {
                    "$inc": {"message_count": 1},
                    "$set": {"last_active": datetime.now(timezone.utc)},
                },
                return_document=False,  # return document BEFORE update
            )
            pre_count = result["message_count"] if result else 0
            logger.info(f"[{session_id}] message_count pre-increment: {pre_count}")
            return pre_count
        except Exception as e:
            logger.error(f"[{session_id}] Failed to increment message count: {e}")
            return 0

    async def get_message_count(self, session_id: str) -> int:
        """Current message count for this session."""
        if not self.is_configured:
            return 0
        try:
            doc = await self._db.sessions.find_one(
                {"session_id": session_id},
                {"message_count": 1}
            )
            return doc["message_count"] if doc else 0
        except Exception as e:
            logger.error(f"[{session_id}] Failed to get message count: {e}")
            return 0

    # ─── Last Messages ─────────────────────────────────────────────────────────

    async def update_last_messages(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Store the most recent user+assistant exchange.
        Used for Phase 2 context (previous turn).
        """
        if not self.is_configured:
            return
        try:
            await self._db.sessions.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "last_messages": {
                            "user": user_message,
                            "assistant": assistant_message,
                        },
                        "last_active": datetime.now(timezone.utc),
                    }
                },
            )
            logger.info(f"[{session_id}] last_messages updated")
        except Exception as e:
            logger.error(f"[{session_id}] Failed to update last_messages: {e}")

    async def get_last_messages(self, session_id: str) -> Optional[Dict[str, str]]:
        """
        Retrieve the most recent user+assistant exchange.
        Returns {"user": str, "assistant": str} or None.
        """
        if not self.is_configured:
            return None
        try:
            doc = await self._db.sessions.find_one(
                {"session_id": session_id},
                {"last_messages": 1}
            )
            if not doc:
                return None
            last = doc.get("last_messages")
            if not last or not isinstance(last, dict):
                return None
            return last if last.get("user") else None
        except Exception as e:
            logger.error(f"[{session_id}] Failed to get last_messages: {e}")
            return None

    # ─── Summaries ─────────────────────────────────────────────────────────────

    async def append_summary(self, session_id: str, summary: dict) -> None:
        """
        Atomically append one exchange summary.
        Uses $push so concurrent background tasks never overwrite each other.
        summary: {"user_intent": str, "bot_response": str, "context": str}
        """
        if not self.is_configured:
            return
        try:
            await self._db.sessions.update_one(
                {"session_id": session_id},
                {
                    "$push": {"summaries": summary},
                    "$set": {"last_active": datetime.now(timezone.utc)},
                },
            )
            logger.info(f"[{session_id}] Summary appended to MongoDB")
        except Exception as e:
            logger.error(f"[{session_id}] Failed to append summary: {e}")

    async def get_summaries(self, session_id: str) -> List[dict]:
        """
        Load all summaries for this session, in chronological order.
        Returns [{"user_intent": str, "bot_response": str, "context": str}, ...]
        """
        if not self.is_configured:
            return []
        try:
            doc = await self._db.sessions.find_one(
                {"session_id": session_id},
                {"summaries": 1}
            )
            if not doc:
                return []
            return doc.get("summaries", [])
        except Exception as e:
            logger.error(f"[{session_id}] Failed to get summaries: {e}")
            return []


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _empty_session(session_id: str) -> dict:
    """Return an in-memory fallback session when MongoDB is not configured."""
    now = datetime.now(timezone.utc)
    return {
        "session_id": session_id,
        "summaries": [],
        "last_messages": {},
        "message_count": 0,
        "created_at": now,
        "last_active": now,
        "is_expired": False,
    }


memory_manager = MemoryManager()
