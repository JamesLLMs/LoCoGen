# LoCoGen: Long Conversation Generation

**Evaluating the Long-Term Memory of Large Language Models**

[![Paper](https://aclanthology.org/2025.findings-acl.1014/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

LoCoGen is an automated pipeline for constructing long-term dialogue datasets to evaluate the long-term memory capabilities of Large Language Models (LLMs). This project implements the methodology described in the paper "Evaluating the Long-Term Memory of Large Language Models".

### Key Features

- **Automated Data Generation**: 5-stage pipeline for creating long-term chronological conversations
- **LOCCO Dataset**: 100 users with 3080 dialogues spanning multiple time periods
- **Memory Evaluation**: Comprehensive framework for testing LLM long-term memory
- **Multiple LLM Support**: Compatible with OpenAI GPT models and local models (InternLM2, Llama, etc.)
- **Modular Architecture**: Clean, well-documented, and easily extensible codebase

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LoCoGen.git
cd LoCoGen
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Basic Usage

```python
from src.api_client import create_client
from src.config import Config

# Initialize LLM client
client = create_client(model_name="gpt-4")

# Generate text
response = client.generate("Your prompt here", max_tokens=500)
print(response)
```

## 📁 Project Structure

```
locogen/
├── src/                          # Source code
│   ├── config.py                 # Configuration management
│   ├── api_client.py             # Unified LLM API client
│   ├── prompts.py                # Prompt templates
│   ├── utils/                    # Utility modules
│   │   ├── json_utils.py         # JSON parsing utilities
│   │   ├── text_utils.py         # Text processing utilities
│   │   └── file_utils.py         # File I/O utilities
│   ├── pipeline/                 # Data generation pipeline
│   │   ├── stage1_character_init.py      # Character initialization
│   │   ├── stage2_diary_generation.py    # Diary generation
│   │   ├── stage3_dialogue_generation.py # Dialogue generation
│   │   ├── stage4_dataset_construction.py # Dataset construction
│   │   └── stage5_question_generation.py  # Question generation
│   └── evaluation/               # Evaluation modules
│       ├── metrics/              # Evaluation metrics (BLEU, ROUGE, etc.)
│       └── consistency_model.py  # Consistency evaluation
├── data/                         # Data directory
│   ├── raw/                      # Raw input data
│   ├── intermediate/             # Intermediate outputs
│   └── final/                    # Final datasets (LOCCO.json, LOCCO_L.json)
├── scripts/                      # Execution scripts
├── notebooks/                    # Jupyter notebooks for analysis
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Configuration

Edit `.env` file to configure:

```bash
# OpenAI API
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# Model settings
DEFAULT_MODEL=gpt-4
MAX_TOKENS=4096
TEMPERATURE=0.7

# Logging
LOG_LEVEL=INFO
```

## 📊 Data Generation Pipeline

The LoCoGen pipeline consists of 5 stages:

### Stage 1: Character Initialization
Generate detailed character profiles with MBTI personality types across 3 time points (1, 3, and 5 years ago).

### Stage 2: Diary Generation
Create temporal diary entries for characters, maintaining consistency and character development.

### Stage 3: Dialogue Generation
Convert diary entries into multi-turn user-chatbot dialogues (3-5 rounds per conversation).

### Stage 4: Dataset Construction
Process dialogues and construct time-split training datasets with cloze-mask tasks.

### Stage 5: Question Generation
Generate memory test questions to evaluate LLM's ability to recall historical information.

## 🎯 Research Questions

This project addresses 6 key research questions:

1. How do LLMs perform in long-term memory tasks?
2. Does memory performance vary with the introduction of new data?
3. Do LLMs exhibit memory preferences similar to humans?
4. Do LLMs experience cognitive load like humans?
5. Do LLMs exhibit a forgetting baseline?
6. Do LLMs achieve permanent memory through replay strategies?

## 📈 Evaluation

The project includes comprehensive evaluation metrics:

- **BLEU**: Bilingual Evaluation Understudy
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation
- **METEOR**: Metric for Evaluation of Translation with Explicit ORdering
- **CIDEr**: Consensus-based Image Description Evaluation
- **Consistency Model**: Custom model for evaluating response consistency

## 🔬 Key Findings

- LLMs can retain past interaction information to a certain extent
- Memory gradually weakens over time
- Rehearsal strategies enhance memory persistence
- LLMs exhibit memory preferences across different information categories
- Excessive rehearsal is not effective for larger models

## 📚 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{locogen2024,
  title={Evaluating the Long-Term Memory of Large Language Models},
  author={Jia, Zixi and Liu, Qinghua and Li, Hexiao and Chen, Yuyan and Liu, Jiqiang},
  journal={arXiv preprint arXiv:2309.16609},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MBTI-S2Conv dataset for character profiles
- OpenAI for GPT models
- Hugging Face for transformer models

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact the authors (see paper for details)

## 🔗 Links

- [Paper (arXiv)](https://arxiv.org/abs/2309.16609)
- [Dataset](https://github.com/JamesLLMs/LoCoGen)
- [Documentation](docs/)

---

**Note**: This is a refactored version of the original LoCoGen project with improved code structure, documentation, and maintainability.
