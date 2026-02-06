"""
Stage 3: Dialogue Generation

This module handles the third stage of the LoCoGen pipeline:
converting diary entries into multi-turn user-chatbot dialogues.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..config import Config
from ..api_client import LLMClient, create_client
from ..prompts import PromptTemplates
from ..utils.json_utils import safe_json_loads, extract_json_from_markdown
from ..utils.file_utils import read_json, write_json

logger = logging.getLogger(__name__)


class DialogueGenerator:
    """
    Dialogue generation pipeline.

    Converts diary entries into multi-turn conversations between
    a user and a chatbot.
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize dialogue generator.

        Args:
            client: LLM client
            input_dir: Directory with diary data (from Stage 2)
            output_dir: Output directory for generated dialogues
        """
        self.client = client or create_client()
        self.input_dir = input_dir or Config.STAGE2_DIR
        self.output_dir = output_dir or Config.STAGE3_DIR

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_dialogue_from_diary(
        self,
        diary_entry: Dict[str, Any],
        character_name: str = "User"
    ) -> List[Dict[str, str]]:
        """
        Generate a multi-turn dialogue from a diary entry.

        Args:
            diary_entry: Diary entry with time and content
            character_name: Name of the character/user

        Returns:
            List of dialogue turns with speaker and message
        """
        logger.info(f"Generating dialogue for diary entry: {diary_entry.get('time')}")

        # Format diary event
        event_str = f"Time: {diary_entry.get('time')}\nEvent: {diary_entry.get('content')}"

        # Example formatted data (you can customize this)
        formatted_data = '{"User": "message", "Chatbot": "response"}'

        # Generate prompt
        prompt = PromptTemplates.format_dialogue_generation_prompt(
            the_event=event_str,
            formatted_data=formatted_data
        )

        # Generate response
        response = self.client.generate(prompt, max_tokens=1000, temperature=0.7)

        # Parse response
        json_str = extract_json_from_markdown(response)
        dialogue_data = safe_json_loads(json_str, default={})

        # Convert to dialogue format
        dialogue = []
        if isinstance(dialogue_data, dict):
            for key, value in dialogue_data.items():
                if 'User' in key or character_name in key:
                    dialogue.append({"speaker": character_name, "message": value})
                elif 'Chatbot' in key or 'Assistant' in key:
                    dialogue.append({"speaker": "Chatbot", "message": value})

        logger.info(f"Generated dialogue with {len(dialogue)} turns")
        return dialogue

    def run(
        self,
        diary_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run Stage 3 pipeline.

        Args:
            diary_file: Path to diary file from Stage 2

        Returns:
            Dictionary with all generated dialogues
        """
        logger.info("Starting Stage 3: Dialogue Generation")

        # Load diary data
        if diary_file is None:
            # Find first diary file
            diary_files = list(self.input_dir.glob("diaries_*.json"))
            if not diary_files:
                raise FileNotFoundError(f"No diary files found in {self.input_dir}")
            diary_file = diary_files[0]

        diary_data = read_json(diary_file)

        # Generate dialogues for each diary entry
        all_dialogues = []
        diaries = diary_data.get('diaries', [])
        character_name = diary_data.get('name', 'User')

        for i, diary in enumerate(diaries):
            logger.info(f"Processing diary {i+1}/{len(diaries)}")

            try:
                dialogue = self.generate_dialogue_from_diary(diary, character_name)

                all_dialogues.append({
                    "diary_time": diary.get('time'),
                    "diary_content": diary.get('content'),
                    "dialogue": dialogue
                })
            except Exception as e:
                logger.error(f"Failed to generate dialogue for diary {i}: {e}")
                continue

        result = {
            "character_id": diary_data.get('character_id'),
            "name": character_name,
            "dialogues": all_dialogues
        }

        # Save results
        output_file = self.output_dir / f"dialogues_{diary_data.get('character_id')}.json"
        write_json(result, output_file)

        logger.info(f"Stage 3 complete. Generated {len(all_dialogues)} dialogues")
        return result


def run_stage3(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_name: str = "gpt-4"
) -> Dict[str, Any]:
    """
    Convenience function to run Stage 3 pipeline.

    Args:
        input_dir: Input directory with Stage 2 results
        output_dir: Output directory
        model_name: LLM model to use

    Returns:
        Generated dialogues
    """
    client = create_client(model_name=model_name)
    generator = DialogueGenerator(
        client=client,
        input_dir=Path(input_dir) if input_dir else None,
        output_dir=Path(output_dir) if output_dir else None
    )
    return generator.run()
