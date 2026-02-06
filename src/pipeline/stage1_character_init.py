"""
Stage 1: Character Initialization

This module handles the first stage of the LoCoGen pipeline:
generating detailed character profiles across multiple time points.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
import random

from ..config import Config
from ..api_client import LLMClient, create_client
from ..prompts import PromptTemplates
from ..utils.json_utils import safe_json_loads, extract_json_from_markdown
from ..utils.file_utils import read_json, write_json

logger = logging.getLogger(__name__)


class CharacterInitializer:
    """
    Character initialization pipeline.

    Generates detailed character profiles at multiple time points
    based on MBTI personality data.
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        input_file: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize character initializer.

        Args:
            client: LLM client (creates default if not provided)
            input_file: Path to MBTI profile dataset
            output_dir: Output directory for generated profiles
        """
        self.client = client or create_client()
        self.input_file = input_file or Config.RAW_DATA_DIR / "mbti_profile_dataset.json"
        self.output_dir = output_dir or Config.STAGE1_DIR

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_mbti_profiles(self) -> List[Dict[str, Any]]:
        """
        Load MBTI character profiles.

        Returns:
            List of character profiles
        """
        logger.info(f"Loading MBTI profiles from: {self.input_file}")

        if not self.input_file.exists():
            raise FileNotFoundError(f"MBTI profile file not found: {self.input_file}")

        data = read_json(self.input_file)

        # Handle different data formats
        if isinstance(data, list):
            profiles = data
        elif isinstance(data, dict) and "profiles" in data:
            profiles = data["profiles"]
        else:
            profiles = [data]

        logger.info(f"Loaded {len(profiles)} character profiles")
        return profiles

    def generate_time_points(
        self,
        base_date: str = "2024-01-01",
        years_back: List[int] = [1, 3, 5]
    ) -> List[str]:
        """
        Generate time points for character descriptions.

        Args:
            base_date: Base date (current time)
            years_back: Years to go back for each time point

        Returns:
            List of date strings (YYYY-MM-DD)
        """
        base = datetime.strptime(base_date, "%Y-%m-%d")
        time_points = []

        for years in years_back:
            date = base - timedelta(days=years * 365)
            time_points.append(date.strftime("%Y-%m-%d"))

        return time_points

    def generate_character_description(
        self,
        character_info: Dict[str, Any],
        time_points: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate character descriptions at multiple time points.

        Args:
            character_info: Character information from MBTI dataset
            time_points: List of time points (generates default if not provided)

        Returns:
            Dictionary with time points as keys and descriptions as values
        """
        if time_points is None:
            time_points = self.generate_time_points()

        logger.info(f"Generating description for character: {character_info.get('name', 'Unknown')}")

        # Format character info for prompt
        char_info_str = json.dumps(character_info, indent=2, ensure_ascii=False)

        # Generate prompt
        prompt = PromptTemplates.format_character_init_prompt(char_info_str)

        # Generate response
        response = self.client.generate(
            prompt,
            max_tokens=2000,
            temperature=0.7
        )

        # Extract and parse JSON
        json_str = extract_json_from_markdown(response)
        descriptions = safe_json_loads(json_str, default={})

        if not descriptions:
            logger.warning(f"Failed to parse character description for: {character_info.get('name')}")
            return {}

        logger.info(f"Successfully generated {len(descriptions)} time point descriptions")
        return descriptions

    def expand_character_timeline(
        self,
        time1_info: Dict[str, Any],
        time2_info: Dict[str, Any],
        n_points: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Expand character timeline by generating intermediate descriptions.

        Args:
            time1_info: Earlier time point information
            time2_info: Later time point information
            n_points: Number of intermediate points to generate

        Returns:
            List of intermediate character descriptions
        """
        logger.info(f"Expanding timeline with {n_points} intermediate points")

        # Format time point information
        time1_str = json.dumps(time1_info, indent=2, ensure_ascii=False)
        time2_str = json.dumps(time2_info, indent=2, ensure_ascii=False)

        # Generate prompt
        prompt = PromptTemplates.format_character_expansion_prompt(
            n=n_points,
            time1_info=time1_str,
            time2_info=time2_str
        )

        # Generate response
        response = self.client.generate(
            prompt,
            max_tokens=2000,
            temperature=0.7
        )

        # Extract and parse JSON
        json_str = extract_json_from_markdown(response)
        intermediate_points = safe_json_loads(json_str, default={})

        if isinstance(intermediate_points, dict):
            # Convert dict to list
            intermediate_points = list(intermediate_points.values())

        logger.info(f"Generated {len(intermediate_points)} intermediate descriptions")
        return intermediate_points

    def process_single_character(
        self,
        character_info: Dict[str, Any],
        expand_timeline: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single character through the initialization pipeline.

        Args:
            character_info: Character information
            expand_timeline: Whether to expand timeline with intermediate points

        Returns:
            Complete character profile with all time points
        """
        # Generate initial descriptions
        descriptions = self.generate_character_description(character_info)

        if expand_timeline and len(descriptions) >= 2:
            # Get first and last time points
            time_points = sorted(descriptions.keys())
            time1 = descriptions[time_points[0]]
            time2 = descriptions[time_points[-1]]

            # Generate intermediate points
            intermediate = self.expand_character_timeline(time1, time2, n_points=2)

            # Merge all descriptions
            # (In practice, you'd need to assign proper dates to intermediate points)
            for i, desc in enumerate(intermediate):
                descriptions[f"intermediate_{i}"] = desc

        return {
            "character_id": character_info.get("id", "unknown"),
            "name": character_info.get("name", "Unknown"),
            "base_info": character_info,
            "timeline": descriptions
        }

    def run(
        self,
        max_characters: Optional[int] = None,
        expand_timeline: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run the complete Stage 1 pipeline.

        Args:
            max_characters: Maximum number of characters to process (None for all)
            expand_timeline: Whether to expand timeline with intermediate points

        Returns:
            List of processed character profiles
        """
        logger.info("Starting Stage 1: Character Initialization")

        # Load MBTI profiles
        profiles = self.load_mbti_profiles()

        if max_characters:
            profiles = profiles[:max_characters]
            logger.info(f"Processing first {max_characters} characters")

        # Process each character
        results = []
        for i, profile in enumerate(profiles, 1):
            logger.info(f"Processing character {i}/{len(profiles)}")

            try:
                result = self.process_single_character(profile, expand_timeline)
                results.append(result)

                # Save intermediate results
                output_file = self.output_dir / f"character_{i:04d}.json"
                write_json(result, output_file)

            except Exception as e:
                logger.error(f"Failed to process character {i}: {e}")
                continue

        # Save all results
        final_output = self.output_dir / "all_characters.json"
        write_json(results, final_output)

        logger.info(f"Stage 1 complete. Processed {len(results)} characters")
        logger.info(f"Results saved to: {final_output}")

        return results


def run_stage1(
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_characters: Optional[int] = None,
    model_name: str = "gpt-4"
) -> List[Dict[str, Any]]:
    """
    Convenience function to run Stage 1 pipeline.

    Args:
        input_file: Path to MBTI profile dataset
        output_dir: Output directory
        max_characters: Maximum number of characters to process
        model_name: LLM model to use

    Returns:
        List of processed character profiles
    """
    # Create client
    client = create_client(model_name=model_name)

    # Create initializer
    initializer = CharacterInitializer(
        client=client,
        input_file=Path(input_file) if input_file else None,
        output_dir=Path(output_dir) if output_dir else None
    )

    # Run pipeline
    return initializer.run(max_characters=max_characters)
