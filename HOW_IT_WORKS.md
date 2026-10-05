# 🧠 How It Works: LangChain Context-Aware AI Assistant

This document explains the internal mechanics of the **LangChain Context-Aware AI Assistant**, focusing on how conversational context is maintained, bounded by a **Sliding Window Buffer**, and composed into **LangChain Expression Language (LCEL)** pipelines.

---

## 1. Why Context-Awareness Matters

In a stateless assistant, every API call starts with zero memory of previous queries. While this is simple, real-world conversational workflows require follow-up questions:
- *"What did I just ask you?"*
- *"Can you rewrite that previous Python function using async/await?"*
- *"Based on the data I shared earlier, what is the best strategy?"*

To answer these, the LLM must receive the conversational context alongside the current query.

---

## 2. The Context Problem: Context Drift & Unbounded Growth

If an application simply appends every user and AI message forever:
1. **Token Cost Explodes**: Every new query repays the cost of all previous messages.
2. **Context Limits**: Models have context length limits (or degrade in quality when context gets excessively large).
3. **Context Drift / Poisoning**: Irrelevant topics discussed 50 turns ago distract the model from current tasks.

### The Solution: Sliding Window Buffer ($K$ Recent Turns)

A **Sliding Window** retains only the $K$ most recent interaction turns (where 1 turn = 1 User query + 1 AI response):

```
Turn 1: [User: Hi, my name is Bishal] -> [AI: Hello Bishal!]          ──┐ (Discarded when N > K)
Turn 2: [User: I live in SF]          -> [AI: SF is great!]           ──┤
Turn 3: [User: I code in Python]      -> [AI: Python is clean!]       ──┼── Active Window
Turn 4: [User: Recommend libraries]   -> [AI: LangChain, PyTorch...]  ──┤   (e.g., K = 3)
Turn 5: [User: What did I say in 3?]  -> [AI: You code in Python!]    ──┘
```

When Turn 5 completes and $K=3$, Turns 1 and 2 are evicted, keeping memory strictly bounded to Turns 3, 4, and 5.

---

## 3. Core Architecture & Modules

### 3.1 `context/manager.py` (ConversationContext)
Inherits from LangChain's official `BaseChatMessageHistory` and manages history using LangChain built-ins:
- **`BaseChatMessageHistory` interface**: Provides native LangChain chat history compatibility (`messages`, `add_user_message`, `add_ai_message`, `clear`).
- **`trim_messages`**: Official LangChain runnable utility for sliding window context pruning (`token_counter=len`, `strategy="last"`, `start_on="human"`).
- **`get_buffer_string`**: LangChain built-in formatting utility for converting message history into human-readable transcripts.
- `set_max_turns(size)`: Dynamically resizes the sliding window and updates the trimmer on `/window <N>`.

### 3.2 `prompts/library.py` & `prompts/builder.py`
Uses LangChain’s official `MessagesPlaceholder`:
```python
CONTEXTUAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", DEFAULT_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
```

### 3.3 `assistant/conversational_ai.py` (LCEL Execution)
```python
# Pure LCEL chain with LangChain trim_messages & RunnablePassthrough
chain = (
    RunnablePassthrough.assign(
        history=itemgetter("history") | trimmer
    )
    | prompt
    | self.llm
    | self.parser
)

# Invoked with both history and current input
response = chain.invoke({
    "history": self.context.get_history_messages(),
    "input": user_input,
})

# Store turn in memory
self.context.add_turn(user_input=user_input, ai_response=str(response))
```

---

## 4. Interactive Session Commands

| Command | Action |
|---|---|
| `/history` | Displays all turns currently held in the sliding window buffer |
| `/clear` | Resets the conversation context buffer |
| `/window <N>` | Changes the sliding window capacity (e.g. `/window 10`) |
| `/system <prompt>` | Updates the active system directive persona |
| `/provider` | Displays active provider, model name, and window size |
| `/help` | Displays available commands banner |
| `/exit` | Exits the application |
