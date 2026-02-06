"""
Unified API client for interacting with various LLM models.

This module provides a unified interface for generating text using different
LLM providers including OpenAI GPT models and local models like InternLM2 and Llama.
"""

import os
import logging
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

from .config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified client for interacting with various LLM models.

    Supports:
    - OpenAI GPT models (GPT-3.5, GPT-4, etc.)
    - Local models (InternLM2, Llama, Mistral, etc.)
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        model_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        gpu_device: str = "0,1"
    ):
        """
        Initialize LLM client.

        Args:
            model_name: Name of the model to use
            model_path: Path to local model (for non-API models)
            api_key: API key for OpenAI (optional, uses config if not provided)
            base_url: Base URL for API (optional, uses config if not provided)
            gpu_device: GPU devices to use (e.g., "0,1")
        """
        self.model_name = model_name
        self.model_path = model_path
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.base_url = base_url or Config.OPENAI_BASE_URL
        self.gpu_device = gpu_device

        # Set CUDA configuration
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = Config.PYTORCH_CUDA_ALLOC_CONF

        self.model = None
        self.tokenizer = None
        self.client = None

        # Initialize model based on type
        if "gpt" in model_name.lower():
            self._init_openai_client()
        else:
            self._init_local_model()

    def _init_openai_client(self) -> None:
        """Initialize OpenAI client."""
        if not self.api_key:
            raise ValueError("API key is required for OpenAI models")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"Initialized OpenAI client for model: {self.model_name}")

    def _init_local_model(self) -> None:
        """Initialize local model."""
        if not self.model_path:
            raise ValueError(f"Model path is required for local model: {self.model_name}")

        logger.info(f"Loading local model: {self.model_name} from {self.model_path}")

        if self.model_name == "InternLM2chat":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            )
            self.model = self.model.eval()

        elif self.model_name == "llama_orca":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )

        elif self.model_name == "maxine7b":
            self.model = transformers.pipeline(
                "text-generation",
                model=self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        else:
            # Generic local model loading
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        logger.info(f"Successfully loaded local model: {self.model_name}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        json_mode: bool = False,
        retry_count: int = 3,
        retry_delay: float = 1.0
    ) -> str:
        """
        Generate text using the configured model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            json_mode: Whether to use JSON response format (OpenAI only)
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds

        Returns:
            Generated text

        Raises:
            Exception: If generation fails after all retries
        """
        for attempt in range(retry_count):
            try:
                if "gpt" in self.model_name.lower():
                    return self._generate_openai(prompt, max_tokens, temperature, json_mode)
                else:
                    return self._generate_local(prompt, max_tokens, temperature)
            except Exception as e:
                logger.warning(
                    f"Generation attempt {attempt + 1}/{retry_count} failed: {str(e)}"
                )
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All generation attempts failed for prompt: {prompt[:100]}...")
                    raise

    def _generate_openai(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool
    ) -> str:
        """Generate text using OpenAI API."""
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        completion = self.client.chat.completions.create(**kwargs)
        result = completion.choices[0].message.content

        logger.debug(f"Generated {len(result)} characters using {self.model_name}")
        return result

    def _generate_local(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate text using local model."""
        if self.model_name == "InternLM2chat":
            outputs = self.model.chat(
                self.tokenizer,
                prompt,
                history=[],
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            return outputs[0].split('\n', 1)[0]

        elif self.model_name == "llama_orca":
            formatted_prompt = f"<|prompter|>{prompt}</s><|assistant|>"
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
            output = self.model.generate(
                **inputs,
                temperature=temperature,
                do_sample=True,
                max_new_tokens=max_tokens
            )
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return generated_text.replace(prompt, "").strip()

        elif self.model_name == "maxine7b":
            outputs = self.model(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature
            )
            generated_text = outputs[0]["generated_text"]
            return generated_text.replace(prompt, "").strip()

        else:
            # Generic generation for other models
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text.replace(prompt, "").strip()

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens

        Note:
            For OpenAI models, this uses tiktoken.
            For local models, uses the model's tokenizer.
        """
        if "gpt" in self.model_name.lower():
            try:
                import tiktoken
                encoding = tiktoken.encoding_for_model(self.model_name)
                return len(encoding.encode(text))
            except Exception as e:
                logger.warning(f"Failed to count tokens with tiktoken: {e}")
                # Fallback: rough estimate
                return len(text.split()) * 1.3

        elif self.tokenizer:
            tokens = self.tokenizer.encode(text)
            return len(tokens)

        else:
            # Fallback: rough estimate
            return len(text.split()) * 1.3


def create_client(
    model_name: str = None,
    api_key: str = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        model_name: Name of the model (uses config default if not provided)
        api_key: API key (uses config default if not provided)
        **kwargs: Additional arguments passed to LLMClient

    Returns:
        Initialized LLMClient instance
    """
    model_name = model_name or Config.DEFAULT_MODEL
    return LLMClient(model_name=model_name, api_key=api_key, **kwargs)
