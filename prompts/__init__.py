"""
prompts package - LangChain Prompt Construction & Templates
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)
from .builder import (
    create_contextual_prompt,
    create_chat_prompt,
    create_prompt_with_instructions,
)
from .library import (
    DEFAULT_SYSTEM_PROMPT,
    CONTEXTUAL_PROMPT,
    CONVERSATIONAL_PROMPT,
    CODE_EXPLAINER_PROMPT,
    TEXT_SUMMARIZER_PROMPT,
    QA_PROMPT,
)

__all__ = [
    "create_contextual_prompt",
    "create_chat_prompt",
    "create_prompt_with_instructions",
    "ChatPromptTemplate",
    "PromptTemplate",
    "MessagesPlaceholder",
    "DEFAULT_SYSTEM_PROMPT",
    "CONTEXTUAL_PROMPT",
    "CONVERSATIONAL_PROMPT",
    "CODE_EXPLAINER_PROMPT",
    "TEXT_SUMMARIZER_PROMPT",
    "QA_PROMPT",
]
