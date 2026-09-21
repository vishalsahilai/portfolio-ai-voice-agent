import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from call.call_state_machine import CallStateMachine
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class Session:
    session_id: str

    state_machine: CallStateMachine = field(init=False)

    conversation_history: List[dict] = field(
        default_factory=list,
        init=False,
    )

    interrupted: bool = field(
        default=False,
        init=False,
    )

    message_count: int = field(
        default=0,
        init=False,
    )

    last_messages: dict = field(
        default_factory=dict,
        init=False,
    )

    def __post_init__(self) -> None:
        self.state_machine = CallStateMachine(
            self.session_id
        )


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create_session(self) -> Session:
        session_id = str(uuid.uuid4())

        session = Session(
            session_id=session_id
        )

        self._sessions[session_id] = session

        logger.info(
            f"Session created: {session_id}"
        )

        return session

    def get_session(
        self,
        session_id: str,
    ) -> Optional[Session]:
        return self._sessions.get(
            session_id
        )

    def end_session(
        self,
        session_id: str,
    ) -> None:
        session = self._sessions.pop(
            session_id,
            None,
        )

        if session:
            logger.info(
                f"Session ended: {session_id}"
            )


session_manager = SessionManager()