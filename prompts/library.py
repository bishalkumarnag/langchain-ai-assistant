# library.py - Predefined LangChain Prompt Templates
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable, and precise AI assistant.\n"
    "Key instructions:\n"
    "1. Provide accurate, clear, and concise responses.\n"
    "2. You have access to conversation history from previous turns. Use this context to answer follow-up questions accurately.\n"
    "3. Maintain a professional, polite, and encouraging tone."
)

CONTEXTUAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", DEFAULT_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

CONVERSATIONAL_PROMPT = CONTEXTUAL_PROMPT

CODE_EXPLAINER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert software engineer and code reviewer."),
    (
        "human",
        "Please analyze and explain the following {language} code snippet:\n\n"
        "```{language}\n{code}\n```\n\n"
        "Format your response with:\n"
        "1. High-Level Summary\n"
        "2. Line-by-Line Breakdown\n"
        "3. Time & Space Complexity\n"
        "4. Potential Edge Cases & Improvements",
    ),
])

TEXT_SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert editor who produces concise summaries."),
    (
        "human",
        "Summarize the following passage in {target_length} format with focus on {focus_area}:\n\n"
        "\"\"\"\n{text}\n\"\"\"\n\n"
        "Summary:",
    ),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a factual Q&A assistant."),
    (
        "human",
        "Context information:\n"
        "\"\"\"\n{context}\n\"\"\"\n\n"
        "Question: {question}\n\n"
        "Provide a direct answer strictly based on the context above.",
    ),
])

CONTEXTUAL_CHAT_PROMPT = CONTEXTUAL_PROMPT
