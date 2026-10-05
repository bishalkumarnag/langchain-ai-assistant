"""
builder.py - LangChain Message Construction & Formatting
"""

from typing import List, Optional
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)


def create_human_message(content: str) -> HumanMessage:
    if not content or not content.strip():
        raise ValueError("Human message content cannot be empty.")
    return HumanMessage(content=content.strip())


def create_system_message(content: str) -> SystemMessage:
    if not content or not content.strip():
        raise ValueError("System message content cannot be empty.")
    return SystemMessage(content=content.strip())


def create_ai_message(content: str) -> AIMessage:
    return AIMessage(content=content)


def format_contextual_turn(
    history_messages: List[BaseMessage],
    user_input: str,
    system_instruction: Optional[str] = None,
) -> List[BaseMessage]:
    """
    Assembles a full list of LangChain messages for a contextual turn:
    [SystemMessage (optional)] + [History Messages] + [Current HumanMessage]
    """
    messages: List[BaseMessage] = []

    if system_instruction and system_instruction.strip():
        messages.append(create_system_message(system_instruction))

    messages.extend(history_messages)
    messages.append(create_human_message(user_input))

    return messages


def format_stateless_turn(
    user_input: str,
    system_instruction: Optional[str] = None,
) -> List[BaseMessage]:
    """Isolated turn without prior context (retained for compatibility)."""
    return format_contextual_turn([], user_input, system_instruction)
