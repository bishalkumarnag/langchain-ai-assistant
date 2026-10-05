"""
manager.py - LangChain Native Sliding Window Conversation History & Context Manager

Maintains conversation history across multiple turns using LangChain's official:
- BaseChatMessageHistory: Abstract base class for chat message history.
- trim_messages: LangChain utility to manage token/message sliding window contexts.
- get_buffer_string: LangChain utility to convert message sequences to string summaries.
- HumanMessage, AIMessage, SystemMessage, BaseMessage: Core message primitives.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
    get_buffer_string,
)


@dataclass
class Turn:
    """Represents a single conversational interaction (User query + AI response)."""
    turn_id: int
    user_input: str
    ai_response: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_messages(self) -> List[BaseMessage]:
        """Converts turn into LangChain HumanMessage and AIMessage."""
        return [
            HumanMessage(content=self.user_input),
            AIMessage(content=self.ai_response),
        ]


class ConversationContext(BaseChatMessageHistory):
    """
    Sliding-window context buffer built on LangChain's BaseChatMessageHistory
    and powered by LangChain's trim_messages for sliding-window pruning.
    
    Attributes:
        max_turns (int): Number of most recent turns (human + AI pairs) to retain.
        system_instruction (Optional[str]): Persona or persistent system prompt.
    """

    def __init__(self, max_turns: int = 5, system_instruction: Optional[str] = None):
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        self.max_turns: int = max_turns
        self.system_instruction: Optional[str] = system_instruction
        self._messages: List[BaseMessage] = []
        self._turn_counter: int = 0
        self._init_trimmer()

    def _init_trimmer(self) -> None:
        """Initializes LangChain's built-in trim_messages runnable for sliding window."""
        # Each turn consists of 1 HumanMessage + 1 AIMessage = 2 messages
        self._trimmer = trim_messages(
            max_tokens=self.max_turns * 2,
            token_counter=len,
            strategy="last",
            start_on="human",
        )

    @property
    def max_turns_allowed(self) -> int:
        return self.max_turns

    def set_max_turns(self, max_turns: int) -> None:
        """Dynamically adjust the sliding window size and re-trim history."""
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        self.max_turns = max_turns
        self._init_trimmer()
        self._trim()

    @property
    def messages(self) -> List[BaseMessage]:
        """All active LangChain messages within the sliding window."""
        return list(self._messages)

    def add_message(self, message: BaseMessage) -> None:
        """Appends a BaseMessage and trims to sliding window using LangChain trim_messages."""
        self._messages.append(message)
        self._trim()

    def _trim(self) -> None:
        """Applies LangChain's official trim_messages to maintain bounded sliding window."""
        if self._messages:
            self._messages = list(self._trimmer.invoke(self._messages))

    def add_turn(self, user_input: str, ai_response: str) -> Turn:
        """
        Records a completed interaction turn into LangChain history and trims with trim_messages.
        """
        self._turn_counter += 1
        turn = Turn(
            turn_id=self._turn_counter,
            user_input=user_input.strip(),
            ai_response=ai_response.strip(),
        )
        self._messages.append(HumanMessage(content=user_input.strip()))
        self._messages.append(AIMessage(content=ai_response.strip()))
        self._trim()
        return turn

    def get_history_messages(self) -> List[BaseMessage]:
        """
        Returns active sliding-window messages as LangChain BaseMessage objects:
        [HumanMessage, AIMessage, HumanMessage, AIMessage, ...]
        """
        return list(self._messages)

    def get_full_context_messages(
        self,
        current_input: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> List[BaseMessage]:
        """
        Assembles complete message list for LLM invocation:
        [SystemMessage(optional)] + [History Messages] + [Current HumanMessage(optional)]
        """
        messages: List[BaseMessage] = []

        # 1. System directive (override or stored)
        active_system = system_instruction or self.system_instruction
        if active_system and active_system.strip():
            messages.append(SystemMessage(content=active_system.strip()))

        # 2. Sliding window history
        messages.extend(self.get_history_messages())

        # 3. Current turn query
        if current_input and current_input.strip():
            messages.append(HumanMessage(content=current_input.strip()))

        return messages

    def clear(self) -> None:
        """Resets all conversation history while preserving system instructions."""
        self._messages.clear()

    @property
    def turn_count(self) -> int:
        """Number of active turns currently in memory window."""
        return len(self._messages) // 2

    @property
    def total_recorded_turns(self) -> int:
        """Cumulative number of turns recorded since session start."""
        return self._turn_counter

    def format_summary(self) -> str:
        """Returns human-readable summary of active context window using LangChain get_buffer_string."""
        if not self._messages:
            return "Context memory is currently empty (0 turns)."

        lines = [
            f"=== Active Context Window ({self.turn_count}/{self.max_turns} turns) ===",
            get_buffer_string(self._messages),
            "==========================================",
        ]
        return "\n".join(lines)
