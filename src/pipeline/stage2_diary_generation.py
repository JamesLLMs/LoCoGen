"""
Stage 2: Diary Generation

This module handles the second stage of the LoCoGen pipeline:
generating diary entries for characters across time periods.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta

from ..config import Config
from ..api_client import LLMClient, create_client
from ..prompts import PromptTemplates
from ..utils.json_utils import safe_json_loads, extract_json_from_markdown
from ..utils.file_utils import read_json, write_json

logger = logging.getLogger(__name__)


class DiaryGenerator:
    """
    Diary generation pipeline.

    Generates coherent diary entries for characters between time points,
    maintaining consistency with character development.
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize diary generator.

        Args:
            client: LLM client
            input_dir: Directory with character descriptions (from Stage 1)
            output_dir: Output directory for generated diaries
        """
        self.client = client or create_client()
        self.input_dir = input_dir or Config.STAGE1_DIR
        self.output_dir = output_dir or Config.STAGE2_DIR

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_diaries_between_timepoints(
        self,
        time1_info: Dict[str, Any],
        time2_info: Dict[str, Any],
        n_entries: int = 5,
        structured_data: Optional[Dict] = None,
        previous_summary: str = "",
        recent_diaries: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Generate diary entries between two time points.

        Args:
            time1_info: Earlier time point information
            time2_info: Later time point information
            n_entries: Number of diary entries to generate
            structured_data: Background structured data
            previous_summary: Summary of previous diaries
            recent_diaries: Recent diary content

        Returns:
            List of diary entries with time and content
        """
        logger.info(f"Generating {n_entries} diary entries between time points")

        # Format inputs
        structured_data_str = str(structured_data) if structured_data else ""
        time1_str = str(time1_info)
        time2_str = str(time2_info)

        # Generate prompt
        prompt = PromptTemplates.format_diary_generation_prompt(
            n=n_entries,
            structured_data_list=structured_data_str,
            time1_describe=time1_str,
            time2_describe=time2_str,
            diaries_summary=previous_summary,
            last_stage_diaries=recent_diaries
        )

        # Generate response
        response = self.client.generate(prompt, max_tokens=2000, temperature=0.7)

        # Parse response
        json_str = extract_json_from_markdown(response)
        diaries_dict = safe_json_loads(json_str, default={})

        # Convert to list format
        diaries = []
        for key, value in diaries_dict.items():
            if isinstance(value, dict) and 'time' in value and 'content' in value:
                diaries.append(value)

        logger.info(f"Generated {len(diaries)} diary entries")
        return diaries

    def summarize_diaries(self, diaries: List[Dict[str, Any]]) -> str:
        """
        Generate summary of diary entries.

        Args:
            diaries: List of diary entries

        Returns:
            Summary text
        """
        # Combine diary contents
        events_content = "\n".join([
            f"{d.get('time', '')}: {d.get('content', '')}"
            for d in diaries
        ])

        prompt = PromptTemplates.DIARY_SUMMARY_TEMPLATE.format(
            events_content=events_content
        )

        summary = self.client.generate(prompt, max_tokens=500, temperature=0.7)
        return summary

    def run(
        self,
        character_file: Optional[Path] = None,
        n_entries_per_period: int = 5
    ) -> Dict[str, Any]:
        """
        Run Stage 2 pipeline for a character.

        Args:
            character_file: Path to character file from Stage 1
            n_entries_per_period: Number of diary entries per time period

        Returns:
            Dictionary with all generated diaries
        """
        logger.info("Starting Stage 2: Diary Generation")

        # Load character data
        if character_file is None:
            character_file = self.input_dir / "all_characters.json"

        character_data = read_json(character_file)

        # Generate diaries for each time period
        all_diaries = []
        timeline = character_data.get('timeline', {})
        time_points = sorted(timeline.keys())

        for i in range(len(time_points) - 1):
            time1 = time_points[i]
            time2 = time_points[i + 1]

            logger.info(f"Generating diaries between {time1} and {time2}")

            diaries = self.generate_diaries_between_timepoints(
                time1_info=timeline[time1],
                time2_info=timeline[time2],
                n_entries=n_entries_per_period
            )

            all_diaries.extend(diaries)

        # Generate summary
        summary = self.summarize_diaries(all_diaries)

        result = {
            "character_id": character_data.get('character_id'),
            "name": character_data.get('name'),
            "diaries": all_diaries,
            "summary": summary
        }

        # Save results
        output_file = self.output_dir / f"diaries_{character_data.get('character_id')}.json"
        write_json(result, output_file)

        logger.info(f"Stage 2 complete. Generated {len(all_diaries)} diary entries")
        return result


def run_stage2(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_name: str = "gpt-4"
) -> Dict[str, Any]:
    """
    Convenience function to run Stage 2 pipeline.

    Args:
        input_dir: Input directory with Stage 1 results
        output_dir: Output directory
        model_name: LLM model to use

    Returns:
        Generated diaries
    """
    client = create_client(model_name=model_name)
    generator = DiaryGenerator(
        client=client,
        input_dir=Path(input_dir) if input_dir else None,
        output_dir=Path(output_dir) if output_dir else None
    )
    return generator.run()
