"""
test_all.py - Comprehensive Unit Tests for LangChain Context-Aware Assistant
"""

import unittest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from context import ConversationContext, Turn
from messages.builder import (
    format_contextual_turn,
    format_stateless_turn,
    create_human_message,
    create_system_message,
    create_ai_message,
)
from prompts.builder import create_contextual_prompt, create_chat_prompt
from assistant.conversational_ai import ContextAwareLangChainAssistant
from parsers import get_string_parser, get_json_parser


class TestConversationContext(unittest.TestCase):
    def test_add_turn_and_retrieval(self):
        from langchain_core.chat_history import BaseChatMessageHistory
        ctx = ConversationContext(max_turns=3)
        self.assertIsInstance(ctx, BaseChatMessageHistory)
        self.assertEqual(ctx.turn_count, 0)

        ctx.add_turn("Hello", "Hi there!")
        self.assertEqual(ctx.turn_count, 1)

        history = ctx.get_history_messages()
        self.assertEqual(len(history), 2)
        self.assertIsInstance(history[0], HumanMessage)
        self.assertIsInstance(history[1], AIMessage)
        self.assertEqual(history[0].content, "Hello")
        self.assertEqual(history[1].content, "Hi there!")

        # Verify LangChain's get_buffer_string format in summary
        summary = ctx.format_summary()
        self.assertIn("Human: Hello", summary)
        self.assertIn("AI: Hi there!", summary)

    def test_sliding_window_trimming(self):
        ctx = ConversationContext(max_turns=2)
        ctx.add_turn("Turn 1", "Reply 1")
        ctx.add_turn("Turn 2", "Reply 2")
        ctx.add_turn("Turn 3", "Reply 3")

        # Must maintain only the 2 most recent turns
        self.assertEqual(ctx.turn_count, 2)
        history = ctx.get_history_messages()
        # Oldest turn (Turn 1) dropped, Turn 2 and Turn 3 remain
        self.assertEqual(history[0].content, "Turn 2")
        self.assertEqual(history[1].content, "Reply 2")
        self.assertEqual(history[2].content, "Turn 3")
        self.assertEqual(history[3].content, "Reply 3")

    def test_clear_context(self):
        ctx = ConversationContext(max_turns=5)
        ctx.add_turn("Query", "Answer")
        self.assertEqual(ctx.turn_count, 1)

        ctx.clear()
        self.assertEqual(ctx.turn_count, 0)
        self.assertEqual(len(ctx.get_history_messages()), 0)

    def test_dynamic_window_resize(self):
        ctx = ConversationContext(max_turns=5)
        for i in range(5):
            ctx.add_turn(f"Q{i}", f"A{i}")
        self.assertEqual(ctx.turn_count, 5)

        ctx.set_max_turns(2)
        self.assertEqual(ctx.turn_count, 2)
        history = ctx.get_history_messages()
        self.assertEqual(history[0].content, "Q3")
        self.assertEqual(history[2].content, "Q4")

    def test_full_context_messages_assembly(self):
        ctx = ConversationContext(max_turns=2, system_instruction="Default Sys")
        ctx.add_turn("User 1", "AI 1")
        messages = ctx.get_full_context_messages(current_input="User 2")

        self.assertEqual(len(messages), 4)  # System + User1 + AI1 + User2
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertEqual(messages[0].content, "Default Sys")
        self.assertEqual(messages[3].content, "User 2")


class TestMessagesAndPrompts(unittest.TestCase):
    def test_message_builders(self):
        h = create_human_message("hello")
        self.assertIsInstance(h, HumanMessage)
        self.assertEqual(h.content, "hello")

        s = create_system_message("sys")
        self.assertIsInstance(s, SystemMessage)

        a = create_ai_message("ai")
        self.assertIsInstance(a, AIMessage)

    def test_contextual_prompt_template(self):
        prompt = create_contextual_prompt("Custom Sys")
        messages = prompt.format_messages(
            history=[HumanMessage(content="Prior Q"), AIMessage(content="Prior A")],
            input="Current Q",
        )
        self.assertEqual(len(messages), 4)  # Sys + Prior Q + Prior A + Current Q
        self.assertEqual(messages[0].content, "Custom Sys")
        self.assertEqual(messages[1].content, "Prior Q")
        self.assertEqual(messages[2].content, "Prior A")
        self.assertEqual(messages[3].content, "Current Q")


class TestContextAwareAssistant(unittest.TestCase):
    def test_assistant_multi_turn_retention(self):
        fake_llm = FakeListChatModel(responses=[
            "Hello Bishal! Nice to meet you.",
            "Your name is Bishal and you are working on AI.",
        ])

        assistant = ContextAwareLangChainAssistant(
            llm=fake_llm,
            window_size=3,
        )

        # Turn 1
        reply1 = assistant.ask("My name is Bishal and I am working on AI.")
        self.assertEqual(reply1, "Hello Bishal! Nice to meet you.")
        self.assertEqual(assistant.context.turn_count, 1)

        # Turn 2
        reply2 = assistant.ask("What is my name and project?")
        self.assertEqual(reply2, "Your name is Bishal and you are working on AI.")
        self.assertEqual(assistant.context.turn_count, 2)

        # Verify history holds both turns
        history = assistant.context.get_history_messages()
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0].content, "My name is Bishal and I am working on AI.")
        self.assertEqual(history[1].content, "Hello Bishal! Nice to meet you.")
        self.assertEqual(history[2].content, "What is my name and project?")
        self.assertEqual(history[3].content, "Your name is Bishal and you are working on AI.")

    def test_clear_context_in_assistant(self):
        fake_llm = FakeListChatModel(responses=["Answer 1", "Answer 2"])
        assistant = ContextAwareLangChainAssistant(llm=fake_llm, window_size=2)

        assistant.ask("Turn 1")
        self.assertEqual(assistant.context.turn_count, 1)

        assistant.clear_context()
        self.assertEqual(assistant.context.turn_count, 0)
        self.assertEqual(len(assistant.context.get_history_messages()), 0)


if __name__ == "__main__":
    unittest.main()
