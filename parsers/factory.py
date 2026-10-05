"""
factory.py - LangChain Output Parser Creation Logic

Provides factory functions to instantiate and configure official LangChain output parsers:
- StrOutputParser: For clean string outputs
- JsonOutputParser: For JSON objects and lists
- PydanticOutputParser: For strongly typed Pydantic models
"""

from typing import Type, TypeVar
from pydantic import BaseModel
from langchain_core.output_parsers import (
    BaseOutputParser,
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)

T = TypeVar("T", bound=BaseModel)


def get_string_parser() -> StrOutputParser:
    """
    Returns an official LangChain StrOutputParser instance for plain text chat.
    """
    return StrOutputParser()


def get_json_parser() -> JsonOutputParser:
    """
    Returns an official LangChain JsonOutputParser instance for JSON responses.
    """
    return JsonOutputParser()


def get_pydantic_parser(pydantic_model: Type[T]) -> PydanticOutputParser[T]:
    """
    Constructs an official LangChain PydanticOutputParser bound to a specific Pydantic schema.

    Args:
        pydantic_model: A Pydantic BaseModel class defining the expected schema.

    Returns:
        PydanticOutputParser: Configured LangChain parser that validates against the schema.
    """
    return PydanticOutputParser(pydantic_object=pydantic_model)
