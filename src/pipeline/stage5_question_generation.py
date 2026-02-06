"""
Stage 5: Question Generation

This module handles the fifth stage of the LoCoGen pipeline:
generating memory test questions from historical dialogues.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import random

from ..config import Config
from ..api_client import LLMClient, create_client
from ..prompts import PromptTemplates
from ..utils.json_utils import safe_json_loads, extract_json_from_markdown
from ..utils.file_utils import read_json, write_json

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    Question generation pipeline.

    Generates memory test questions based on historical dialogues
    to evaluate LLM's long-term memory capabilities.
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize question generator.

        Args:
            client: LLM client
            input_dir: Directory with dataset (from Stage 4)
            output_dir: Output directory for generated questions
        """
        self.client = client or create_client()
        self.input_dir = input_dir or Config.STAGE4_DIR
        self.output_dir = output_dir or Config.FINAL_DATA_DIR

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_dialogue_history(
        self,
        examples: List[Dict[str, Any]],
        max_turns: int = 10
    ) -> str:
        """
        Format dialogue history for prompt.

        Args:
            examples: List of dialogue examples
            max_turns: Maximum number of turns to include

        Returns:
            Formatted history string
        """
        history_parts = []

        for example in examples[-max_turns:]:
            timestamp = example.get('timestamp', '')
            user_msg = example.get('user_message', '')
            bot_msg = example.get('bot_response', '')

            history_parts.append(f"[{timestamp}]")
            history_parts.append(f"User: {user_msg}")
            history_parts.append(f"Chatbot: {bot_msg}")
            history_parts.append("")

        return "\n".join(history_parts)

    def generate_qa_pair(
        self,
        history_examples: List[Dict[str, Any]],
        current_time: str
    ) -> Optional[Dict[str, str]]:
        """
        Generate a question-answer pair from dialogue history.

        Args:
            history_examples: Historical dialogue examples
            current_time: Current timestamp

        Returns:
            Dictionary with question and answer, or None if generation fails
        """
        logger.info(f"Generating QA pair for time: {current_time}")

        # Format history
        history_str = self.format_dialogue_history(history_examples)

        # Generate prompt
        prompt = PromptTemplates.format_qa_generation_prompt(
            current_time=current_time,
            history_conversation=history_str
        )

        # Generate response
        response = self.client.generate(prompt, max_tokens=500, temperature=0.7)

        # Parse response
        json_str = extract_json_from_markdown(response)
        qa_data = safe_json_loads(json_str, default={})

        if 'User' in qa_data and 'Chatbot' in qa_data:
            return {
                "question": qa_data['User'],
                "answer": qa_data['Chatbot'],
                "timestamp": current_time,
                "history_length": len(history_examples)
            }

        logger.warning("Failed to parse QA pair from response")
        return None

    def generate_questions_for_dataset(
        self,
        dataset: Dict[str, Any],
        questions_per_period: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate questions for entire dataset.

        Args:
            dataset: Dataset from Stage 4
            questions_per_period: Number of questions per time period

        Returns:
            List of QA pairs
        """
        logger.info("Generating questions for dataset")

        all_qa_pairs = []
        periods = dataset.get('periods', [])

        for period_idx, period in enumerate(periods):
            logger.info(f"Processing period {period_idx + 1}/{len(periods)}")

            examples = period.get('examples', [])

            if not examples:
                logger.warning(f"No examples in period {period_idx}")
                continue

            # Generate questions for this period
            for _ in range(questions_per_period):
                # Select random subset of history
                if len(examples) > 5:
                    history_sample = random.sample(examples[:-1], min(5, len(examples) - 1))
                    history_sample.sort(key=lambda x: x.get('timestamp', ''))
                else:
                    history_sample = examples[:-1]

                # Use last example's time as current time
                current_time = examples[-1].get('timestamp', '')

                try:
                    qa_pair = self.generate_qa_pair(history_sample, current_time)
                    if qa_pair:
                        qa_pair['period_id'] = period_idx
                        all_qa_pairs.append(qa_pair)
                except Exception as e:
                    logger.error(f"Failed to generate QA pair: {e}")
                    continue

        logger.info(f"Generated {len(all_qa_pairs)} QA pairs")
        return all_qa_pairs

    def run(
        self,
        dataset_file: Optional[Path] = None,
        questions_per_period: int = 5
    ) -> Dict[str, Any]:
        """
        Run Stage 5 pipeline.

        Args:
            dataset_file: Path to dataset file from Stage 4
            questions_per_period: Number of questions per period

        Returns:
            Dictionary with generated questions
        """
        logger.info("Starting Stage 5: Question Generation")

        # Find dataset file
        if dataset_file is None:
            dataset_files = list(self.input_dir.glob("dataset_*.json"))
            if not dataset_files:
                raise FileNotFoundError(f"No dataset files found in {self.input_dir}")
            dataset_file = dataset_files[0]

        # Load dataset
        dataset = read_json(dataset_file)

        # Generate questions
        qa_pairs = self.generate_questions_for_dataset(dataset, questions_per_period)

        result = {
            "character_id": dataset.get('character_id'),
            "name": dataset.get('name'),
            "n_questions": len(qa_pairs),
            "qa_pairs": qa_pairs
        }

        # Save results
        output_file = self.output_dir / f"LOCCO_{dataset.get('character_id')}.json"
        write_json(result, output_file)

        logger.info(f"Stage 5 complete. Generated {len(qa_pairs)} questions")
        return result


def run_stage5(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    questions_per_period: int = 5,
    model_name: str = "gpt-4"
) -> Dict[str, Any]:
    """
    Convenience function to run Stage 5 pipeline.

    Args:
        input_dir: Input directory with Stage 4 results
        output_dir: Output directory
        questions_per_period: Number of questions per period
        model_name: LLM model to use

    Returns:
        Generated questions
    """
    client = create_client(model_name=model_name)
    generator = QuestionGenerator(
        client=client,
        input_dir=Path(input_dir) if input_dir else None,
        output_dir=Path(output_dir) if output_dir else None
    )
    return generator.run(questions_per_period=questions_per_period)
