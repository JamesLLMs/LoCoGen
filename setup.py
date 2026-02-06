"""
Setup script for LoCoGen package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README_NEW.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="locogen",
    version="1.0.0",
    author="Zixi Jia, Qinghua Liu, Hexiao Li, Yuyan Chen, Jiqiang Liu",
    author_email="",
    description="Long Conversation Generation for Evaluating LLM Long-Term Memory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JamesLLMs/LoCoGen",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "tiktoken>=0.5.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "nltk>=3.8",
        "rouge-score>=0.1.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
)
