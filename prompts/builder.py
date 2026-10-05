"""
builder.py - LangChain Prompt Construction Logic
"""

from typing import Dict, Any, Optional
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)
from prompts.library import DEFAULT_SYSTEM_PROMPT


def create_contextual_prompt(
    system_instruction: Optional[str] = None,
    user_input_placeholder: str = "{input}",
) -> ChatPromptTemplate:
    """
    Builds a LangChain ChatPromptTemplate with a system prompt,
    a history MessagesPlaceholder, and a human input placeholder.
    """
    instruction = system_instruction if system_instruction else DEFAULT_SYSTEM_PROMPT
    return ChatPromptTemplate.from_messages([
        ("system", instruction),
        MessagesPlaceholder(variable_name="history"),
        ("human", user_input_placeholder),
    ])


def create_chat_prompt(
    system_instruction: str,
    user_input_placeholder: str = "{input}",
) -> ChatPromptTemplate:
    """
    Constructs a standard LangChain ChatPromptTemplate.
    """
    return ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", user_input_placeholder),
    ])


def create_prompt_with_instructions(
    system_instruction: str,
    format_instructions: str,
    user_input_placeholder: str = "{input}",
) -> ChatPromptTemplate:
    """
    Builds a LangChain ChatPromptTemplate with output schema format instructions.
    """
    augmented_system = f"{system_instruction}\n\nFormat Instructions:\n{format_instructions}"
    return create_chat_prompt(
        system_instruction=augmented_system,
        user_input_placeholder=user_input_placeholder,
    )
