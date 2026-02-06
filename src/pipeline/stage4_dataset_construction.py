"""
Stage 4: Dataset Construction

This module handles the fourth stage of the LoCoGen pipeline:
constructing training datasets from dialogues with time-based splits.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from ..config import Config
from ..utils.file_utils import read_json, write_json, write_jsonl

logger = logging.getLogger(__name__)


class DatasetConstructor:
    """
    Dataset construction pipeline.

    Processes dialogues and constructs time-split training datasets
    for evaluating long-term memory.
    """

    def __init__(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize dataset constructor.

        Args:
            input_dir: Directory with dialogue data (from Stage 3)
            output_dir: Output directory for constructed datasets
        """
        self.input_dir = input_dir or Config.STAGE3_DIR
        self.output_dir = output_dir or Config.STAGE4_DIR

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def split_dialogues_by_time(
        self,
        dialogues: List[Dict[str, Any]],
        n_periods: int = 6
    ) -> List[List[Dict[str, Any]]]:
        """
        Split dialogues into time periods.

        Args:
            dialogues: List of dialogues with timestamps
            n_periods: Number of time periods to split into

        Returns:
            List of dialogue lists, one per time period
        """
        logger.info(f"Splitting {len(dialogues)} dialogues into {n_periods} periods")

        # Sort dialogues by time
        sorted_dialogues = sorted(
            dialogues,
            key=lambda x: x.get('diary_time', '1900-01-01')
        )

        # Split into equal periods
        period_size = len(sorted_dialogues) // n_periods
        periods = []

        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = start_idx + period_size if i < n_periods - 1 else len(sorted_dialogues)
            periods.append(sorted_dialogues[start_idx:end_idx])

        logger.info(f"Created {len(periods)} time periods")
        return periods

    def create_training_examples(
        self,
        dialogue: Dict[str, Any],
        include_history: bool = True,
        history_dialogues: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create training examples from a dialogue.

        Args:
            dialogue: Dialogue data
            include_history: Whether to include historical context
            history_dialogues: Previous dialogues for context

        Returns:
            List of training examples
        """
        examples = []
        dialogue_turns = dialogue.get('dialogue', [])

        # Build conversation history
        history = []
        if include_history and history_dialogues:
            for hist_dlg in history_dialogues[-3:]:  # Last 3 dialogues
                for turn in hist_dlg.get('dialogue', []):
                    history.append({
                        "speaker": turn.get('speaker'),
                        "message": turn.get('message')
                    })

        # Create examples from current dialogue
        for i in range(0, len(dialogue_turns), 2):
            if i + 1 < len(dialogue_turns):
                user_turn = dialogue_turns[i]
                bot_turn = dialogue_turns[i + 1]

                example = {
                    "timestamp": dialogue.get('diary_time'),
                    "history": history.copy(),
                    "user_message": user_turn.get('message'),
                    "bot_response": bot_turn.get('message'),
                    "diary_context": dialogue.get('diary_content')
                }

                examples.append(example)

                # Add to history for next turn
                history.append({"speaker": user_turn.get('speaker'), "message": user_turn.get('message')})
                history.append({"speaker": bot_turn.get('speaker'), "message": bot_turn.get('message')})

        return examples

    def construct_dataset(
        self,
        dialogue_file: Path,
        n_periods: int = 6
    ) -> Dict[str, Any]:
        """
        Construct complete dataset with time splits.

        Args:
            dialogue_file: Path to dialogue file from Stage 3
            n_periods: Number of time periods

        Returns:
            Dictionary with dataset splits
        """
        logger.info("Constructing dataset with time-based splits")

        # Load dialogues
        dialogue_data = read_json(dialogue_file)
        dialogues = dialogue_data.get('dialogues', [])

        # Split by time
        time_periods = self.split_dialogues_by_time(dialogues, n_periods)

        # Create training examples for each period
        dataset = {
            "character_id": dialogue_data.get('character_id'),
            "name": dialogue_data.get('name'),
            "periods": []
        }

        cumulative_history = []

        for period_idx, period_dialogues in enumerate(time_periods):
            logger.info(f"Processing period {period_idx + 1}/{n_periods}")

            period_examples = []

            for dialogue in period_dialogues:
                examples = self.create_training_examples(
                    dialogue,
                    include_history=True,
                    history_dialogues=cumulative_history
                )
                period_examples.extend(examples)
                cumulative_history.append(dialogue)

            dataset["periods"].append({
                "period_id": period_idx,
                "n_dialogues": len(period_dialogues),
                "n_examples": len(period_examples),
                "examples": period_examples
            })

        return dataset

    def run(
        self,
        dialogue_file: Optional[Path] = None,
        n_periods: int = 6
    ) -> Dict[str, Any]:
        """
        Run Stage 4 pipeline.

        Args:
            dialogue_file: Path to dialogue file from Stage 3
            n_periods: Number of time periods to split into

        Returns:
            Constructed dataset
        """
        logger.info("Starting Stage 4: Dataset Construction")

        # Find dialogue file
        if dialogue_file is None:
            dialogue_files = list(self.input_dir.glob("dialogues_*.json"))
            if not dialogue_files:
                raise FileNotFoundError(f"No dialogue files found in {self.input_dir}")
            dialogue_file = dialogue_files[0]

        # Construct dataset
        dataset = self.construct_dataset(dialogue_file, n_periods)

        # Save results
        output_file = self.output_dir / f"dataset_{dataset.get('character_id')}.json"
        write_json(dataset, output_file)

        # Also save as JSONL for easier processing
        all_examples = []
        for period in dataset['periods']:
            all_examples.extend(period['examples'])

        jsonl_file = self.output_dir / f"dataset_{dataset.get('character_id')}.jsonl"
        write_jsonl(all_examples, jsonl_file)

        logger.info(f"Stage 4 complete. Created dataset with {len(all_examples)} examples")
        return dataset


def run_stage4(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    n_periods: int = 6
) -> Dict[str, Any]:
    """
    Convenience function to run Stage 4 pipeline.

    Args:
        input_dir: Input directory with Stage 3 results
        output_dir: Output directory
        n_periods: Number of time periods

    Returns:
        Constructed dataset
    """
    constructor = DatasetConstructor(
        input_dir=Path(input_dir) if input_dir else None,
        output_dir=Path(output_dir) if output_dir else None
    )
    return constructor.run(n_periods=n_periods)
