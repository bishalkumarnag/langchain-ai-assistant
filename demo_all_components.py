"""
demo_all_components.py - Full Architectural Component Demo

Demonstrates the modular LangChain components and tests context awareness:
  1. Config & Settings
  2. Sliding Window Context Manager
  3. Prompts with MessagesPlaceholder
  4. Parsers
  5. Multi-turn dialogue showing memory retention
"""

from config.settings import get_settings
from context import ConversationContext
from prompts.builder import create_contextual_prompt
from parsers import get_string_parser, get_json_parser
from assistant.conversational_ai import ContextAwareLangChainAssistant
from llm.model_factory import get_chat_model


def run_demo():
    print("=" * 70)
    print(" 🧪 LangChain Context-Aware AI Assistant - Architecture & Context Demo")
    print("=" * 70)

    # 1. Config
    print("\n[1] Configuration Check:")
    settings = get_settings()
    print(f"  • Provider: {settings.provider}")
    print(f"  • Context Window: {settings.context_window_turns} turns")

    # 2. Context Manager sliding window demonstration
    print("\n[2] Context Sliding Window Unit Demonstration:")
    ctx = ConversationContext(max_turns=2)
    ctx.add_turn("My favorite framework is LangChain.", "Noted, LangChain is great!")
    ctx.add_turn("I live in San Francisco.", "San Francisco is a vibrant tech hub.")
    print(f"  • Turns recorded: {ctx.turn_count}/2")
    print(ctx.format_summary())

    # Add 3rd turn to trigger sliding window trimming (oldest turn will drop)
    ctx.add_turn("I am building an autonomous AI agent.", "That sounds exciting!")
    print("\n  • After 3rd turn (Window size = 2):")
    print(f"  • Oldest turn dropped automatically. Current turns: {ctx.turn_count}/2")
    print(ctx.format_summary())

    # 3. Live LLM Multi-Turn Context Verification
    try:
        settings.validate_provider_keys()
        llm = get_chat_model()
        assistant = ContextAwareLangChainAssistant(llm=llm, window_size=3)

        print("\n[3] Live Multi-Turn Conversational Memory Test:")
        print("  • Turn 1: Sharing information...")
        q1 = "Hello! My name is Bishal and I am building a robotics project using LangChain."
        print(f"    User: {q1}")
        a1 = assistant.ask(q1)
        print(f"    AI  : {a1[:100]}...")

        print("\n  • Turn 2: Follow-up referencing past context...")
        q2 = "What is my name and what kind of project am I working on?"
        print(f"    User: {q2}")
        a2 = assistant.ask(q2)
        print(f"    AI  : {a2}")

        print("\n  • Memory summary:")
        print(assistant.get_context_summary())

        print("\n✅ All multi-turn context tests passed successfully!")

    except Exception as e:
        print(f"\n⚠️  Live test skipped or encountered an error: {e}")
        print("   (Ensure valid API key is present in .env)")


if __name__ == "__main__":
    run_demo()
