"""
Text processing utilities for LoCoGen project.

This module provides functions for text manipulation and processing.
"""

import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def replace_speaker_numbers(text: str, user_name: str) -> str:
    """
    Replace numbered speakers with actual names in dialogue text.

    This function replaces odd numbers (1, 3, 5, 7) with the user name
    and even numbers (2, 4, 6, 8) with 'Chatbot'.

    Args:
        text: Text containing numbered speakers
        user_name: Name to use for user (odd numbers)

    Returns:
        Text with speakers replaced

    Example:
        >>> replace_speaker_numbers('"1": "Hello"', "Alice")
        'Alice: "Hello"'
    """
    # Define replacement rules for quoted numbers
    replacements = {
        r'"1"': f'"{user_name}"',
        r'"3"': f'"{user_name}"',
        r'"5"': f'"{user_name}"',
        r'"7"': f'"{user_name}"',
        r'"2"': '"Chatbot"',
        r'"4"': '"Chatbot"',
        r'"6"': '"Chatbot"',
        r'"8"': '"Chatbot"'
    }

    # Apply replacements for double-quoted numbers
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # Define replacement rules for single-quoted numbers
    replacements_single = {
        r"'1'": f"'{user_name}'",
        r"'3'": f"'{user_name}'",
        r"'5'": f"'{user_name}'",
        r"'7'": f"'{user_name}'",
        r"'2'": "'Chatbot'",
        r"'4'": "'Chatbot'",
        r"'6'": "'Chatbot'",
        r"'8'": "'Chatbot'"
    }

    # Apply replacements for single-quoted numbers
    for pattern, replacement in replacements_single.items():
        text = re.sub(pattern, replacement, text)

    return text


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated (default: "...")

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    try:
        from nltk import sent_tokenize
        return sent_tokenize(text)
    except ImportError:
        logger.warning("NLTK not available, using simple sentence splitting")
        # Fallback: simple splitting on period, exclamation, question mark
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]


def extract_quoted_text(text: str) -> List[str]:
    """
    Extract all quoted text from string.

    Args:
        text: Input text

    Returns:
        List of quoted strings
    """
    # Find all text within double quotes
    pattern = r'"([^"]*)"'
    matches = re.findall(pattern, text)
    return matches


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    - Replace multiple spaces with single space
    - Replace tabs with spaces
    - Remove leading/trailing whitespace

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Replace tabs with spaces
    text = text.replace('\t', ' ')

    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def count_words(text: str) -> int:
    """
    Count words in text.

    Args:
        text: Input text

    Returns:
        Number of words
    """
    return len(text.split())


def extract_json_from_code_block(text: str) -> str:
    """
    Extract JSON from markdown code block.

    Args:
        text: Text containing markdown code block

    Returns:
        Extracted JSON string
    """
    # Try to find JSON in markdown code block
    pattern = r'```(?:json)?\s*\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()

    return text


def remove_special_characters(text: str, keep_spaces: bool = True) -> str:
    """
    Remove special characters from text.

    Args:
        text: Input text
        keep_spaces: Whether to keep spaces (default: True)

    Returns:
        Text with special characters removed
    """
    if keep_spaces:
        # Keep alphanumeric and spaces
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    else:
        # Keep only alphanumeric
        return re.sub(r'[^a-zA-Z0-9]', '', text)


def format_dialogue(speaker: str, message: str) -> str:
    """
    Format a dialogue turn.

    Args:
        speaker: Speaker name
        message: Message content

    Returns:
        Formatted dialogue string

    Example:
        >>> format_dialogue("Alice", "Hello!")
        'Alice: Hello!'
    """
    return f"{speaker}: {message}"


def parse_dialogue(dialogue_text: str) -> List[Dict[str, str]]:
    """
    Parse dialogue text into structured format.

    Args:
        dialogue_text: Text containing dialogue

    Returns:
        List of dialogue turns with speaker and message

    Example:
        >>> parse_dialogue("Alice: Hello!\\nBob: Hi there!")
        [{'speaker': 'Alice', 'message': 'Hello!'}, {'speaker': 'Bob', 'message': 'Hi there!'}]
    """
    turns = []
    lines = dialogue_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if ':' in line:
            speaker, message = line.split(':', 1)
            turns.append({
                'speaker': speaker.strip(),
                'message': message.strip()
            })

    return turns
