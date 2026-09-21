from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)

class CallState(str, Enum):
    LISTENING = "LISTENING"   # waiting for / receiving user audio
    THINKING = "THINKING"     # STT done, waiting on LLM (+ RAG/memory) response
    SPEAKING = "SPEAKING"     # streaming TTS audio back to the user

# States the machine is allowed to move to, from each current state.
_ALLOWED_TRANSITIONS = {
    CallState.LISTENING: {CallState.THINKING},
    CallState.THINKING: {CallState.SPEAKING, CallState.LISTENING},  # LISTENING = e.g. LLM error, retry
    CallState.SPEAKING: {CallState.LISTENING},  # includes barge-in interrupts
}

class InvalidTransition(Exception):
    pass

class CallStateMachine:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._state = CallState.LISTENING

    @property
    def state(self) -> CallState:
        return self._state
 
    def transition(self, new_state: CallState) -> None:
        if new_state not in _ALLOWED_TRANSITIONS[self._state]:
            raise InvalidTransition(
                f"[{self.session_id}] Cannot go {self._state} -> {new_state}"
            )
        logger.info(f"[{self.session_id}] {self._state} -> {new_state}")
        self._state = new_state

    def interrupt(self) -> None:
        """Force-return to LISTENING from any state — used for barge-in."""
        if self._state != CallState.LISTENING:
            logger.info(f"[{self.session_id}] INTERRUPT: {self._state} -> LISTENING")
            self._state = CallState.LISTENING
