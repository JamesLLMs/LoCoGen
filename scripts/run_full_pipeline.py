"""
Complete Pipeline Runner

This script runs the entire LoCoGen pipeline from Stage 1 to Stage 5.
"""

import sys
import logging
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.api_client import create_client
from src.pipeline.stage1_character_init import CharacterInitializer
from src.pipeline.stage2_diary_generation import DiaryGenerator
from src.pipeline.stage3_dialogue_generation import DialogueGenerator
from src.pipeline.stage4_dataset_construction import DatasetConstructor
from src.pipeline.stage5_question_generation import QuestionGenerator

logger = logging.getLogger(__name__)


def run_full_pipeline(
    model_name: str = "gpt-4",
    max_characters: int = None,
    n_diary_entries: int = 5,
    n_time_periods: int = 6,
    questions_per_period: int = 5
):
    """
    Run the complete LoCoGen pipeline.

    Args:
        model_name: LLM model to use
        max_characters: Maximum number of characters to process
        n_diary_entries: Number of diary entries per time period
        n_time_periods: Number of time periods for dataset split
        questions_per_period: Number of questions per period

    Returns:
        Dictionary with results from all stages
    """
    logger.info("=" * 60)
    logger.info("Starting Complete LoCoGen Pipeline")
    logger.info("=" * 60)

    # Create LLM client
    client = create_client(model_name=model_name)

    results = {}

    # Stage 1: Character Initialization
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1: Character Initialization")
    logger.info("=" * 60)

    stage1 = CharacterInitializer(client=client)
    results['stage1'] = stage1.run(max_characters=max_characters)

    logger.info(f"✓ Stage 1 complete: {len(results['stage1'])} characters initialized")

    # Stage 2: Diary Generation
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: Diary Generation")
    logger.info("=" * 60)

    stage2 = DiaryGenerator(client=client)
    results['stage2'] = stage2.run(n_entries_per_period=n_diary_entries)

    logger.info(f"✓ Stage 2 complete: {len(results['stage2']['diaries'])} diary entries generated")

    # Stage 3: Dialogue Generation
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: Dialogue Generation")
    logger.info("=" * 60)

    stage3 = DialogueGenerator(client=client)
    results['stage3'] = stage3.run()

    logger.info(f"✓ Stage 3 complete: {len(results['stage3']['dialogues'])} dialogues generated")

    # Stage 4: Dataset Construction
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 4: Dataset Construction")
    logger.info("=" * 60)

    stage4 = DatasetConstructor()
    results['stage4'] = stage4.run(n_periods=n_time_periods)

    total_examples = sum(p['n_examples'] for p in results['stage4']['periods'])
    logger.info(f"✓ Stage 4 complete: {total_examples} training examples created")

    # Stage 5: Question Generation
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 5: Question Generation")
    logger.info("=" * 60)

    stage5 = QuestionGenerator(client=client)
    results['stage5'] = stage5.run(questions_per_period=questions_per_period)

    logger.info(f"✓ Stage 5 complete: {results['stage5']['n_questions']} questions generated")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Characters processed: {len(results['stage1'])}")
    logger.info(f"Diary entries: {len(results['stage2']['diaries'])}")
    logger.info(f"Dialogues: {len(results['stage3']['dialogues'])}")
    logger.info(f"Training examples: {total_examples}")
    logger.info(f"Test questions: {results['stage5']['n_questions']}")
    logger.info("\nResults saved to:")
    logger.info(f"  - Stage 1: {Config.STAGE1_DIR}")
    logger.info(f"  - Stage 2: {Config.STAGE2_DIR}")
    logger.info(f"  - Stage 3: {Config.STAGE3_DIR}")
    logger.info(f"  - Stage 4: {Config.STAGE4_DIR}")
    logger.info(f"  - Stage 5 (LOCCO): {Config.FINAL_DATA_DIR}")
    logger.info("=" * 60)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the complete LoCoGen pipeline"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="LLM model to use (default: gpt-4)"
    )

    parser.add_argument(
        "--max-characters",
        type=int,
        default=None,
        help="Maximum number of characters to process (default: all)"
    )

    parser.add_argument(
        "--diary-entries",
        type=int,
        default=5,
        help="Number of diary entries per time period (default: 5)"
    )

    parser.add_argument(
        "--time-periods",
        type=int,
        default=6,
        help="Number of time periods for dataset split (default: 6)"
    )

    parser.add_argument(
        "--questions-per-period",
        type=int,
        default=5,
        help="Number of questions per period (default: 5)"
    )

    args = parser.parse_args()

    try:
        results = run_full_pipeline(
            model_name=args.model,
            max_characters=args.max_characters,
            n_diary_entries=args.diary_entries,
            n_time_periods=args.time_periods,
            questions_per_period=args.questions_per_period
        )

        print("\n✓ Pipeline completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n✗ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
