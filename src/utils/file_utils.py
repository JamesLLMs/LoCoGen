"""
File I/O utilities for LoCoGen project.

This module provides convenient functions for reading and writing
various file formats used in the project.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def read_json(file_path: Union[str, Path], encoding: str = 'utf-8') -> Any:
    """
    Read JSON file.

    Args:
        file_path: Path to JSON file
        encoding: File encoding (default: utf-8)

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.debug(f"Reading JSON from: {file_path}")

    with open(file_path, 'r', encoding=encoding) as f:
        data = json.load(f)

    logger.info(f"Successfully read JSON from: {file_path}")
    return data


def write_json(
    data: Any,
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """
    Write data to JSON file.

    Args:
        data: Data to write
        file_path: Path to output file
        encoding: File encoding (default: utf-8)
        indent: JSON indentation (default: 2)
        ensure_ascii: Whether to escape non-ASCII characters (default: False)
    """
    file_path = Path(file_path)

    # Create parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Writing JSON to: {file_path}")

    with open(file_path, 'w', encoding=encoding) as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

    logger.info(f"Successfully wrote JSON to: {file_path}")


def read_jsonl(file_path: Union[str, Path], encoding: str = 'utf-8') -> List[Dict]:
    """
    Read JSONL (JSON Lines) file.

    Args:
        file_path: Path to JSONL file
        encoding: File encoding (default: utf-8)

    Returns:
        List of parsed JSON objects
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.debug(f"Reading JSONL from: {file_path}")

    data = []
    with open(file_path, 'r', encoding=encoding) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")

    logger.info(f"Successfully read {len(data)} lines from: {file_path}")
    return data


def write_jsonl(
    data: List[Dict],
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    ensure_ascii: bool = False
) -> None:
    """
    Write data to JSONL (JSON Lines) file.

    Args:
        data: List of dictionaries to write
        file_path: Path to output file
        encoding: File encoding (default: utf-8)
        ensure_ascii: Whether to escape non-ASCII characters (default: False)
    """
    file_path = Path(file_path)

    # Create parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Writing JSONL to: {file_path}")

    with open(file_path, 'w', encoding=encoding) as f:
        for item in data:
            json_str = json.dumps(item, ensure_ascii=ensure_ascii)
            f.write(json_str + '\n')

    logger.info(f"Successfully wrote {len(data)} lines to: {file_path}")


def read_text(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """
    Read text file.

    Args:
        file_path: Path to text file
        encoding: File encoding (default: utf-8)

    Returns:
        File contents as string
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.debug(f"Reading text from: {file_path}")

    with open(file_path, 'r', encoding=encoding) as f:
        content = f.read()

    logger.info(f"Successfully read text from: {file_path}")
    return content


def write_text(
    content: str,
    file_path: Union[str, Path],
    encoding: str = 'utf-8'
) -> None:
    """
    Write text to file.

    Args:
        content: Text content to write
        file_path: Path to output file
        encoding: File encoding (default: utf-8)
    """
    file_path = Path(file_path)

    # Create parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Writing text to: {file_path}")

    with open(file_path, 'w', encoding=encoding) as f:
        f.write(content)

    logger.info(f"Successfully wrote text to: {file_path}")


def append_to_jsonl(
    item: Dict,
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    ensure_ascii: bool = False
) -> None:
    """
    Append a single item to JSONL file.

    Args:
        item: Dictionary to append
        file_path: Path to JSONL file
        encoding: File encoding (default: utf-8)
        ensure_ascii: Whether to escape non-ASCII characters (default: False)
    """
    file_path = Path(file_path)

    # Create parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'a', encoding=encoding) as f:
        json_str = json.dumps(item, ensure_ascii=ensure_ascii)
        f.write(json_str + '\n')


def file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if file exists.

    Args:
        file_path: Path to check

    Returns:
        True if file exists, False otherwise
    """
    return Path(file_path).exists()


def ensure_directory(dir_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        dir_path: Directory path

    Returns:
        Path object for the directory
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
