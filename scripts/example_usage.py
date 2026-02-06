"""
Example script demonstrating the usage of refactored LoCoGen modules.

This script shows how to use the new modular structure to:
1. Initialize an LLM client
2. Generate text
3. Parse JSON responses
4. Save results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.api_client import create_client
from src.utils.json_utils import extract_json_from_markdown, safe_json_loads
from src.utils.file_utils import write_json, read_json
from src.utils.text_utils import clean_text


def example_basic_generation():
    """Example: Basic text generation."""
    print("=" * 60)
    print("Example 1: Basic Text Generation")
    print("=" * 60)

    # Create client (uses config defaults)
    client = create_client()

    # Generate text
    prompt = "Write a short story about a robot learning to paint."
    response = client.generate(prompt, max_tokens=200, temperature=0.7)

    print(f"\nPrompt: {prompt}")
    print(f"\nResponse:\n{response}")
    print()


def example_json_generation():
    """Example: Generate and parse JSON."""
    print("=" * 60)
    print("Example 2: JSON Generation and Parsing")
    print("=" * 60)

    client = create_client()

    # Generate JSON
    prompt = """Generate a character profile in JSON format with the following fields:
    - name: character name
    - age: character age
    - personality: brief personality description
    - hobbies: list of hobbies

    Return only valid JSON."""

    response = client.generate(prompt, max_tokens=300, temperature=0.7)

    print(f"\nRaw response:\n{response}\n")

    # Extract JSON from markdown if present
    json_str = extract_json_from_markdown(response)
    print(f"Extracted JSON:\n{json_str}\n")

    # Parse JSON
    data = safe_json_loads(json_str, default={})
    print(f"Parsed data:\n{data}\n")


def example_file_operations():
    """Example: File I/O operations."""
    print("=" * 60)
    print("Example 3: File I/O Operations")
    print("=" * 60)

    # Sample data
    data = {
        "character": "Alice",
        "age": 25,
        "personality": "Creative and curious",
        "hobbies": ["painting", "reading", "chess"]
    }

    # Write JSON
    output_file = Config.INTERMEDIATE_DATA_DIR / "example_output.json"
    write_json(data, output_file)
    print(f"\nWrote data to: {output_file}")

    # Read JSON
    loaded_data = read_json(output_file)
    print(f"\nLoaded data: {loaded_data}")


def example_text_processing():
    """Example: Text processing utilities."""
    print("=" * 60)
    print("Example 4: Text Processing")
    print("=" * 60)

    # Sample text with extra whitespace
    text = "  This   is  a   sample   text   with   extra   spaces.  "

    # Clean text
    cleaned = clean_text(text)
    print(f"\nOriginal: '{text}'")
    print(f"Cleaned: '{cleaned}'")

    # Replace speaker numbers
    from src.utils.text_utils import replace_speaker_numbers

    dialogue = '{"1": "Hello!", "2": "Hi there!", "1": "How are you?"}'
    replaced = replace_speaker_numbers(dialogue, "Alice")
    print(f"\nOriginal dialogue: {dialogue}")
    print(f"Replaced: {replaced}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LoCoGen Refactored Modules - Usage Examples")
    print("=" * 60 + "\n")

    # Note: These examples require API keys to be configured
    print("Note: Make sure to configure your API keys in .env file\n")

    try:
        # Run examples that don't require API calls
        example_file_operations()
        example_text_processing()

        # Uncomment to run API examples (requires valid API key)
        # example_basic_generation()
        # example_json_generation()

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Configured your .env file with API keys")
        print("2. Installed all required dependencies")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
