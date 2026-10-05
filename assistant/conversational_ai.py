"""
conversational_ai.py - LangChain Context-Aware Conversational Assistant

Orchestrates:
  - LLM: from llm.model_factory import get_chat_model
  - Context: from context import ConversationContext (implements BaseChatMessageHistory)
  - Prompts: from prompts.builder import create_contextual_prompt
  - Parsers: from parsers import get_string_parser, BaseOutputParser
  - Trimming: LangChain official trim_messages runnable

Features:
  - Context retention across multi-turn interactions.
  - Bounded memory via LangChain's official trim_messages sliding window.
  - Declarative LangChain Expression Language (LCEL) execution pipeline:
      RunnablePassthrough.assign(history=itemgetter("history") | trimmer) | prompt | llm | parser
"""

from operator import itemgetter
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.messages import trim_messages

from llm.model_factory import get_chat_model
from context import ConversationContext
from prompts.builder import create_contextual_prompt
from prompts.library import CONTEXTUAL_CHAT_PROMPT
from parsers import get_string_parser
from config.settings import get_settings


class ContextAwareLangChainAssistant:
    """
    Context-aware conversational assistant maintaining dialogue history
    via a LangChain native sliding-window memory buffer and LCEL pipeline.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        context: Optional[ConversationContext] = None,
        prompt: Optional[ChatPromptTemplate] = None,
        parser: Optional[BaseOutputParser] = None,
        window_size: Optional[int] = None,
    ):
        settings = get_settings()
        self.llm: BaseChatModel = llm or get_chat_model()

        # Context manager implementing LangChain's BaseChatMessageHistory
        max_turns = window_size or settings.context_window_turns
        self.context: ConversationContext = context or ConversationContext(max_turns=max_turns)

        self.prompt: ChatPromptTemplate = prompt or CONTEXTUAL_CHAT_PROMPT
        self.parser: BaseOutputParser = parser or get_string_parser()

        self._rebuild_chain()

    def _rebuild_chain(self) -> None:
        """Constructs the pure LangChain LCEL pipeline with built-in message trimming."""
        self.trimmer = trim_messages(
            max_tokens=self.context.max_turns * 2,
            token_counter=len,
            strategy="last",
            start_on="human",
        )
        self.chain: Runnable = (
            RunnablePassthrough.assign(
                history=itemgetter("history") | self.trimmer
            )
            | self.prompt
            | self.llm
            | self.parser
        )

    def ask(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        parser: Optional[BaseOutputParser] = None,
    ) -> str:
        """
        Executes a conversational turn:
          1. Prepares prompt and parser (or defaults).
          2. Invokes LCEL chain with active history and user_input.
          3. Appends turn to LangChain history buffer with sliding-window enforcement.
        """
        active_prompt = (
            create_contextual_prompt(system_instruction=system_prompt)
            if system_prompt
            else self.prompt
        )
        active_parser = parser or self.parser

        chain = (
            RunnablePassthrough.assign(
                history=itemgetter("history") | self.trimmer
            )
            | active_prompt
            | self.llm
            | active_parser
        )

        response = chain.invoke({
            "history": self.context.get_history_messages(),
            "input": user_input,
        })

        str_response = str(response)
        self.context.add_turn(user_input=user_input, ai_response=str_response)
        return str_response

    def clear_context(self) -> None:
        """Resets conversation history memory."""
        self.context.clear()

    def get_context_summary(self) -> str:
        """Returns a formatted view of current turns in memory using LangChain get_buffer_string."""
        return self.context.format_summary()

    def set_window_size(self, size: int) -> None:
        """Updates the sliding window size and refreshes LCEL trimming runnable."""
        self.context.set_max_turns(size)
        self._rebuild_chain()


# Alias for backward compatibility
LangChainAssistant = ContextAwareLangChainAssistant
