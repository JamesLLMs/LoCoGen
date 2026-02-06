"""
JSON parsing and extraction utilities.

This module provides functions for extracting and parsing JSON data
from various sources, including handling malformed JSON.
"""

import json
import re
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def extract_outermost_braces_content(text: str) -> Optional[str]:
    """
    Extract content within the outermost curly braces.

    Args:
        text: Input string containing JSON

    Returns:
        Content within outermost braces, or None if not found

    Example:
        >>> extract_outermost_braces_content('Some text {"key": "value"} more text')
        '{"key": "value"}'
    """
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def extract_content(json_string: str) -> str:
    """
    Extract content from OpenAI API response format.

    This function handles the nested structure of OpenAI API responses
    and extracts the actual content from the messages field.

    Args:
        json_string: JSON string from API response

    Returns:
        Extracted content string

    Example:
        >>> response = '{"messages": [{"content": "Hello"}]}'
        >>> extract_content(response)
        'Hello'
    """
    try:
        data = json.loads(json_string, strict=False)

        # Check for 'messages' key
        if 'messages' in data:
            messages = data['messages']
            if messages and 'content' in messages[0]:
                generated_json = messages[0]['content']

                # Handle list content
                if isinstance(generated_json, list):
                    generated_json = generated_json[0]

                # Try to parse as JSON
                try:
                    parsed_data = json.loads(generated_json, strict=False)
                    if isinstance(parsed_data, list) and len(parsed_data) == 1:
                        if isinstance(parsed_data[0], dict):
                            generated_json = parsed_data[0]
                except (json.JSONDecodeError, TypeError):
                    pass

                # Return as string
                if not isinstance(generated_json, str):
                    return json.dumps(generated_json)
                else:
                    # Clean up brackets
                    if isinstance(generated_json, str):
                        generated_json = generated_json.replace("[", "").replace("]", "")
                    return generated_json
            else:
                return json_string

        return json_string

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return json_string


def extract_complete_entries(json_str: str) -> Dict[str, Any]:
    """
    Extract complete entries from potentially malformed JSON.

    This function attempts to parse JSON even if it's incomplete or malformed
    by progressively trying to parse smaller portions of the string.

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Dictionary of successfully parsed entries

    Example:
        >>> malformed = '{"key1": "value1", "key2": "val'
        >>> extract_complete_entries(malformed)
        {'key1': 'value1'}
    """
    complete_entries = {}

    try:
        # Try to parse the full string
        data = json.loads(json_str, strict=False)

        # Handle nested response structure
        if "response" in data:
            json_str = extract_outermost_braces_content(data["response"])
            if json_str:
                data = json.loads(json_str, strict=False)

        # Extract all entries
        for key, value in data.items():
            complete_entries[key] = value

    except json.JSONDecodeError as e:
        # Try to salvage partial JSON
        logger.warning(f"JSON decode error at position {e.pos}, attempting partial parse")

        last_pos = e.pos
        for pos in range(last_pos):
            try_pos = last_pos - pos
            if try_pos > 0:
                try:
                    # Try to close the JSON and parse
                    partial_json = json_str[:try_pos] + "}"
                    data = json.loads(partial_json, strict=False)

                    # Handle nested response structure
                    if "response" in data:
                        json_str = extract_outermost_braces_content(data["response"])
                        if json_str:
                            data = json.loads(json_str, strict=False)

                    # Extract all entries
                    for key, value in data.items():
                        complete_entries[key] = value

                    logger.info(f"Successfully parsed partial JSON up to position {try_pos}")
                    break

                except json.JSONDecodeError:
                    continue

    return complete_entries


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely load JSON with fallback to default value.

    Args:
        json_str: JSON string to parse
        default: Default value to return on error (default: None)

    Returns:
        Parsed JSON data or default value

    Example:
        >>> safe_json_loads('{"key": "value"}')
        {'key': 'value'}
        >>> safe_json_loads('invalid json', default={})
        {}
    """
    try:
        return json.loads(json_str, strict=False)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return default


def extract_json_from_markdown(text: str) -> Optional[str]:
    """
    Extract JSON from markdown code blocks.

    Args:
        text: Text containing markdown code block with JSON

    Returns:
        Extracted JSON string, or None if not found

    Example:
        >>> text = '```json\\n{"key": "value"}\\n```'
        >>> extract_json_from_markdown(text)
        '{"key": "value"}'
    """
    # Try to find JSON in markdown code block
    pattern = r'```(?:json)?\s*\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()

    # If no code block, try to find JSON directly
    return extract_outermost_braces_content(text)


def validate_json_structure(data: Dict, required_keys: list) -> bool:
    """
    Validate that JSON data contains required keys.

    Args:
        data: Dictionary to validate
        required_keys: List of required key names

    Returns:
        True if all required keys are present, False otherwise

    Example:
        >>> data = {"name": "John", "age": 30}
        >>> validate_json_structure(data, ["name", "age"])
        True
        >>> validate_json_structure(data, ["name", "email"])
        False
    """
    if not isinstance(data, dict):
        return False

    for key in required_keys:
        if key not in data:
            logger.warning(f"Missing required key: {key}")
            return False

    return True
