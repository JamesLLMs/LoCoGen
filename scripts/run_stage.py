"""
Run Individual Pipeline Stage

This script allows running a single stage of the LoCoGen pipeline.
"""

import sys
import logging
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.pipeline.stage1_character_init import run_stage1
from src.pipeline.stage2_diary_generation import run_stage2
from src.pipeline.stage3_dialogue_generation import run_stage3
from src.pipeline.stage4_dataset_construction import run_stage4
from src.pipeline.stage5_question_generation import run_stage5

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run a single stage of the LoCoGen pipeline"
    )

    parser.add_argument(
        "stage",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Stage number to run (1-5)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="LLM model to use (default: gpt-4)"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory (uses default if not specified)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (uses default if not specified)"
    )

    # Stage-specific arguments
    parser.add_argument(
        "--max-characters",
        type=int,
        default=None,
        help="[Stage 1] Maximum number of characters to process"
    )

    parser.add_argument(
        "--time-periods",
        type=int,
        default=6,
        help="[Stage 4] Number of time periods for dataset split"
    )

    parser.add_argument(
        "--questions-per-period",
        type=int,
        default=5,
        help="[Stage 5] Number of questions per period"
    )

    args = parser.parse_args()

    try:
        logger.info(f"Running Stage {args.stage}")

        if args.stage == 1:
            result = run_stage1(
                input_file=args.input_dir,
                output_dir=args.output_dir,
                max_characters=args.max_characters,
                model_name=args.model
            )
            print(f"\n✓ Stage 1 complete: {len(result)} characters processed")

        elif args.stage == 2:
            result = run_stage2(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                model_name=args.model
            )
            print(f"\n✓ Stage 2 complete: {len(result['diaries'])} diary entries generated")

        elif args.stage == 3:
            result = run_stage3(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                model_name=args.model
            )
            print(f"\n✓ Stage 3 complete: {len(result['dialogues'])} dialogues generated")

        elif args.stage == 4:
            result = run_stage4(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                n_periods=args.time_periods
            )
            total_examples = sum(p['n_examples'] for p in result['periods'])
            print(f"\n✓ Stage 4 complete: {total_examples} training examples created")

        elif args.stage == 5:
            result = run_stage5(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                questions_per_period=args.questions_per_period,
                model_name=args.model
            )
            print(f"\n✓ Stage 5 complete: {result['n_questions']} questions generated")

        return 0

    except Exception as e:
        logger.error(f"Stage {args.stage} failed: {e}", exc_info=True)
        print(f"\n✗ Stage {args.stage} failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
