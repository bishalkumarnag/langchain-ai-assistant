# 🤖 LangChain Context-Aware AI Assistant

> Production-ready, context-aware conversational AI assistant built with **LangChain (LCEL)**, featuring a **Sliding Window Conversation Context Buffer**, modular prompt management, structured output parsing, and multi-provider LLM support (Google Gemini & OpenAI).

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3%2B-green.svg)](https://python.langchain.com/)

---

## 🌟 Highlights

- **🧠 Sliding Window Context Memory**: Preserves the last $N$ interaction turns (configurable via `.env` or dynamically via `/window <N>`) so the assistant remembers past conversation context and answers follow-up queries seamlessly.
- **⚡ Token-Efficient & Bounded**: Automatically trims older turns outside the sliding window to prevent unbounded context growth, context drift, and excessive token costs.
- **🔗 Pure LangChain LCEL**: Leverages `ChatPromptTemplate` with `MessagesPlaceholder(variable_name="history") | llm | parser` for standard, declarative execution.
- **🔄 Multi-Provider Support**: Seamlessly toggle between **Google Gemini** (`gemini-3.5-flash-lite`, etc.) and **OpenAI** (`gpt-4o-mini`, etc.) via `.env`.
- **🛠️ Interactive CLI Controls**:
  - `/history` — Inspect active conversational turns stored in memory.
  - `/clear` — Wipe conversation history to start a fresh topic.
  - `/window <N>` — Dynamically expand or shrink the sliding window size.
  - `/system <prompt>` — Update system directives on the fly.
  - `/provider` — View active provider and model details.
- **🧪 Comprehensive Test Suite**: 100% passing unit tests covering context management, sliding window trimming, dynamic window resizing, and LCEL chaining.

---

## 🏛️ Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │              User Input Query                │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │        context.ConversationContext           │
                      │  • Stores history turns (Human + AI)         │
                      │  • Sliding window trimming: last N turns     │
                      └──────────────────────┬───────────────────────┘
                                             │
             ┌───────────────────────────────┴───────────────────────────────┐
             │                                                               │
             ▼                                                               ▼
┌─────────────────────────┐                                     ┌─────────────────────────┐
│ prompts.builder         │                                     │ messages.builder        │
│ create_contextual_prompt│                                     │ format_contextual_turn  │
│  [SystemDirective]      │                                     │  [SystemMessage]        │
│  [MessagesPlaceholder]  │                                     │  [Recent Turns (H & AI)]│
│  [HumanInputPlaceholder]│                                     │  [Current HumanMessage] │
└────────────┬────────────┘                                     └────────────┬────────────┘
             │                                                               │
             └───────────────────────────────┬───────────────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     LCEL Pipeline         │
                               │   prompt | llm | parser   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   llm.model_factory       │
                               │  Google Gemini / OpenAI   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     parsers.factory       │
                               │ Str / JSON / Markdown     │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │    Context Update Step    │
                               │  context.add_turn(Q, A)   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │       Assistant Output    │
                               └───────────────────────────┘
```

---

## 📂 Project Structure

```bash
langchain-ai-assistant/
├── .env.example              # Template for environment variables and context size
├── .gitignore                # Ignores credentials, virtual environments, and cache
├── README.md                 # Project overview and documentation
├── HOW_IT_WORKS.md           # Deep dive into sliding window context architecture
├── requirements.txt          # Python dependencies
├── main.py                   # Interactive context-aware CLI assistant
├── demo_all_components.py    # Architectural and multi-turn context demonstration
│
├── config/                   # Configuration package
│   ├── __init__.py
│   └── settings.py           # Loads .env, validates API keys & context window size
│
├── context/                  # Sliding window conversational memory package
│   ├── __init__.py           # Exports ConversationContext and Turn
│   └── manager.py            # Sliding window buffer, turn tracking, and trimming
│
├── llm/                      # LLM Provider abstraction
│   ├── __init__.py
│   └── model_factory.py      # Instantiates Google Gemini or OpenAI Chat Models
│
├── prompts/                  # Prompt construction & templates
│   ├── __init__.py
│   ├── builder.py            # Dynamic prompt templates with MessagesPlaceholder
│   └── library.py            # Predefined system directives and templates
│
├── messages/                 # LangChain message construction
│   ├── __init__.py
│   └── builder.py            # HumanMessage, AIMessage, SystemMessage builders
│
├── parsers/                  # Output parsing package
│   ├── __init__.py
│   └── factory.py            # String, JSON, Markdown, and Custom parsers
│
└── tests/                    # Unit tests
    ├── __init__.py
    └── test_all.py           # Comprehensive tests for context, window, and LCEL
```

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10+
- An API Key from [Google AI Studio](https://aistudio.google.com/) or [OpenAI](https://platform.openai.com/)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/bishalkumarnag/langchain-ai-assistant.git
cd langchain-ai-assistant

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and provide your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
LLM_PROVIDER=google
GEMINI_API_KEY=AIzaSy...
GEMINI_DEFAULT_MODEL=gemini-3.5-flash-lite

# Context sliding window size (number of recent turns to retain)
CONTEXT_WINDOW_TURNS=5
```

---

## 💻 Running the Assistant

### Interactive Terminal Assistant
```bash
python main.py
```

Example dialogue:
```text
========================================================================
 🤖 LangChain Context-Aware AI Assistant (Sliding Window Memory)
========================================================================
 Provider       : GOOGLE
 Model ID       : gemini-3.5-flash-lite
 Context Window : Last 5 turns (Sliding Window Buffer)
------------------------------------------------------------------------

You > Hi! My name is Bishal and I am building a robotics project using LangChain.

AI (1/5 turns in context) > Hello Bishal! That sounds like an exciting project. Combining robotics with LangChain opens up great possibilities for agentic reasoning and natural language control.

You > What is my name and what project am I working on?

AI (2/5 turns in context) > Your name is Bishal, and you are working on a robotics project using LangChain!

You > /history

=== Active Context Window (2/5 turns) ===
[Turn #1]
  User: Hi! My name is Bishal and I am building a robotics project using LangChain.
  AI  : Hello Bishal! That sounds like an exciting project...
[Turn #2]
  User: What is my name and what project am I working on?
  AI  : Your name is Bishal, and you are working on a robotics project using LangChain!
==========================================
```

### Component & Context Demo
```bash
python demo_all_components.py
```

### Run Unit Tests
```bash
python -m unittest discover tests
```

---

## 🧠 How the Sliding Window Context Works

1. **Native LangChain Base**: `ConversationContext` subclasses LangChain's official `BaseChatMessageHistory`, storing standard `HumanMessage` and `AIMessage` pairs.
2. **Official Message Trimming**: Powered by LangChain's built-in `trim_messages` runnable (`token_counter=len`, `strategy="last"`, `start_on="human"`), ensuring only the latest $N$ turns are retained.
3. **Automatic & Dynamic Trimming**: When turn $N+1$ arrives, older turns are pruned by `trim_messages`, keeping token consumption fixed and predictable. Dynamic resizing with `/window <N>` updates the trimmer on the fly.
4. **LCEL Integration**: Composed into LangChain Expression Language pipelines via `RunnablePassthrough.assign(history=itemgetter("history") | trimmer) | prompt | llm | parser` and `MessagesPlaceholder(variable_name="history")`.
5. **Standard Formatting**: Summaries and logs use LangChain's built-in `get_buffer_string` utility.

---

## 👤 Author

**Bishal Kumar Nag**
- GitHub: [@bishalkumarnag](https://github.com/bishalkumarnag)
- LinkedIn: [Bishal Kumar Nag](https://www.linkedin.com/in/bishal-kumar-nag/)

