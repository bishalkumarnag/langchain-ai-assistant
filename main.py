"""
main.py - Interactive CLI for LangChain Context-Aware Assistant
"""

import sys
from config.settings import get_settings
from llm.model_factory import get_chat_model
from assistant.conversational_ai import ContextAwareLangChainAssistant


def print_banner(window_size: int, active_model: str, provider: str):
    print("=" * 72)
    print(" 🤖 LangChain Context-Aware AI Assistant (Sliding Window Memory)")
    print("=" * 72)
    print(f" Provider       : {provider.upper()}")
    print(f" Model ID       : {active_model}")
    print(f" Context Window : Last {window_size} turns (Sliding Window Buffer)")
    print("-" * 72)
    print(" Interactive Commands:")
    print("   /history        - View active conversation history in context memory")
    print("   /clear          - Reset context memory (start fresh conversation)")
    print("   /window <N>     - Change sliding window size (e.g. /window 10)")
    print("   /system <prompt>- Update system persona/directive")
    print("   /provider       - Display model & provider details")
    print("   /help           - Display this help banner")
    print("   /exit or /quit  - Exit the assistant")
    print("=" * 72)


def main():
    settings = get_settings()

    try:
        settings.validate_provider_keys()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}\n")
        sys.exit(1)

    try:
        llm = get_chat_model()
    except Exception as e:
        print(f"\n❌ Failed to initialize Chat Model: {e}\n")
        sys.exit(1)

    assistant = ContextAwareLangChainAssistant(
        llm=llm,
        window_size=settings.context_window_turns,
    )

    active_model = (
        settings.gemini_model if settings.provider == "google" else settings.openai_model
    )
    print_banner(assistant.context.max_turns, active_model, settings.provider)

    system_prompt = None

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                print("Goodbye!")
                break

            elif user_input.lower() == "/help":
                print_banner(assistant.context.max_turns, active_model, settings.provider)
                continue

            elif user_input.lower() == "/history":
                print(f"\n{assistant.get_context_summary()}")
                continue

            elif user_input.lower() == "/clear":
                assistant.clear_context()
                print("🧹 Context memory cleared! Starting fresh conversation.")
                continue

            elif user_input.lower().startswith("/window "):
                arg = user_input[8:].strip()
                try:
                    new_size = int(arg)
                    assistant.set_window_size(new_size)
                    print(f"📐 Context sliding window updated to last {new_size} turns.")
                except ValueError:
                    print("❌ Please provide a valid integer for window size (e.g. /window 8).")
                continue

            elif user_input.lower() == "/provider":
                print(f"Active Provider : {settings.provider.upper()}")
                print(f"Chat Model Class: {type(llm).__name__}")
                print(f"Active Model ID : {active_model}")
                print(f"Context Window  : {assistant.context.max_turns} turns")
                print("💡 To switch provider, set LLM_PROVIDER=openai in .env")
                continue

            elif user_input.lower().startswith("/system "):
                system_prompt = user_input[8:].strip()
                print(f"System directive updated to:\n\"{system_prompt}\"")
                continue

            print(f"\nAI ({assistant.context.turn_count}/{assistant.context.max_turns} turns in context) > ", end="", flush=True)
            response = assistant.ask(user_input, system_prompt=system_prompt)
            print(response)

        except KeyboardInterrupt:
            print("\n\nSession terminated by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")


if __name__ == "__main__":
    main()
