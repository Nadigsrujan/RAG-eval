"""
LLM Generator — wraps Groq Cloud API and HuggingFace models for RAG answer generation.

Supports:
  1. Groq Cloud API (ultra-fast inference: llama-3.3-70b-versatile, llama-3.1-8b-instant, etc.)
  2. Local HuggingFace causal language models (Qwen3, Mistral, Llama)
  3. Local HuggingFace encoder-decoder models (FLAN-T5)

Records token counts, latency, and reasoning traces for observability.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Known Groq model identifiers
GROQ_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "qwen-2.5-32b",
    "deepseek-r1-distill-llama-70b",
}

# Known model aliases for convenience
MODEL_ALIASES = {
    "qwen3-4b-instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen/qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen/qwen3-4b-instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "flan-t5": "google/flan-t5-base",
    "flan-t5-base": "google/flan-t5-base",
    "google/flan-t5-base": "google/flan-t5-base",
    "llama-3.3": "llama-3.3-70b-versatile",
    "llama-3.1": "llama-3.1-8b-instant",
    "llama-8b": "llama-3.1-8b-instant",
    "llama-70b": "llama-3.3-70b-versatile",
}


@dataclass
class GenerationResult:
    """Complete generation output with diagnostics."""

    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_name: str
    prompt: str = ""
    metadata: dict = field(default_factory=dict)


class Generator:
    """
    LLM wrapper for RAG answer generation.

    Supports Groq cloud inference and local HuggingFace transformers pipelines.

    Args:
        model_name: Model identifier or alias (Groq or HuggingFace).
        max_new_tokens: Maximum tokens to generate.
        max_input_tokens: Maximum input tokens (truncates longer prompts).
        temperature: Sampling temperature (0 = greedy).
        device: Device to run local HF models on. None = auto-detect.
        provider: Provider to use ("groq", "huggingface", or auto-detect).
        api_key: Optional Groq API key (defaults to GROQ_API_KEY environment variable).
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        max_new_tokens: int = 500,
        max_input_tokens: int = 1024,
        temperature: float = 0.1,
        device: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
    ):
        clean_name = model_name.strip()
        if clean_name.lower().startswith("groq/"):
            clean_name = clean_name[5:]
            provider = "groq"

        self.model_name = MODEL_ALIASES.get(clean_name.lower(), clean_name)
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")

        # Determine provider
        if provider:
            self.provider = provider.lower()
        elif self.model_name in GROQ_MODELS or (
            self._api_key
            and not self.model_name.startswith("google/")
            and not self.model_name.startswith("Qwen/")
            and not self.model_name.startswith("cross-encoder/")
        ):
            self.provider = "groq"
        else:
            self.provider = "huggingface"

        self._groq_client = None
        self._pipeline = None
        self._tokenizer = None
        self._device = device
        self._task = None

    @property
    def is_groq(self) -> bool:
        """True if using Groq cloud inference."""
        return self.provider == "groq"

    @property
    def pipeline(self):
        """Lazy-load the generation pipeline or Groq client."""
        if self.is_groq:
            if self._groq_client is None:
                from groq import Groq
                api_key = self._api_key or os.environ.get("GROQ_API_KEY")
                if not api_key:
                    logger.warning("GROQ_API_KEY environment variable is not set!")
                else:
                    self._groq_client = Groq(api_key=api_key)
            return self._groq_client

        if self._pipeline is None:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
                pipeline,
            )

            logger.info("Loading local generation model: %s", self.model_name)

            # Auto-detect best device if not explicitly set
            if self._device is not None:
                device_arg = self._device
            elif torch.cuda.is_available():
                device_arg = "cuda"
            elif torch.backends.mps.is_available():
                device_arg = "mps"
            else:
                device_arg = "cpu"

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Inspect model architecture from config
            try:
                config = AutoConfig.from_pretrained(self.model_name)
                archs = [a.lower() for a in getattr(config, "architectures", [])]
                is_seq2seq = any("seq2seq" in a or "conditionalgeneration" in a for a in archs)
            except Exception:
                is_seq2seq = False

            if is_seq2seq:
                self._task = "text2text-generation"
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            else:
                self._task = "text-generation"
                # Select optimal precision for memory efficiency
                if device_arg in ("cuda", "mps"):
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float32

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                )

            # Ensure pad_token is set for causal LMs
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            self._pipeline = pipeline(
                self._task,
                model=model,
                tokenizer=self._tokenizer,
                device=device_arg,
            )
            logger.info("Generation model loaded (task=%s, device=%s)", self._task, device_arg)
        return self._pipeline

    @property
    def tokenizer(self):
        """Access the tokenizer (loads pipeline if needed)."""
        if self.is_groq:
            return None
        if self._tokenizer is None:
            _ = self.pipeline  # triggers lazy load
        return self._tokenizer

    def _truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt to max_input_tokens, preserving head and tail."""
        if self.is_groq:
            char_limit = self.max_input_tokens * 4
            if len(prompt) <= char_limit:
                return prompt
            logger.warning("Prompt truncated from %d to %d chars", len(prompt), char_limit)
            keep_tail = 512
            keep_head = char_limit - keep_tail
            return prompt[:keep_head] + "\n...[truncated]...\n" + prompt[-keep_tail:]

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(input_ids) <= self.max_input_tokens:
            return prompt

        logger.warning(
            "Prompt truncated from %d to %d tokens",
            len(input_ids),
            self.max_input_tokens,
        )
        keep_tail = 128
        keep_head = self.max_input_tokens - keep_tail
        head_ids = input_ids[:keep_head]
        tail_ids = input_ids[-keep_tail:]
        truncated_ids = head_ids + tail_ids
        return self.tokenizer.decode(truncated_ids, skip_special_tokens=True)

    def _generate_groq(self, prompt: str) -> GenerationResult:
        """Generate an answer using Groq cloud API."""
        from groq import Groq

        api_key = self._api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please set the GROQ_API_KEY environment variable "
                "or add GROQ_API_KEY=gsk_... in your .env file."
            )

        client = self._groq_client or Groq(api_key=api_key)
        self._groq_client = client

        start = time.time()
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        latency_ms = (time.time() - start) * 1000

        raw_answer = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else len(prompt) // 4
        output_tokens = response.usage.completion_tokens if response.usage else len(raw_answer) // 4

        metadata = {}
        # Parse reasoning tags if model produced thinking blocks (<think>...</think>)
        if "<think>" in raw_answer and "</think>" in raw_answer:
            think_start = raw_answer.find("<think>") + len("<think>")
            think_end = raw_answer.find("</think>")
            metadata["reasoning"] = raw_answer[think_start:think_end].strip()
            answer = raw_answer[think_end + len("</think>"):].strip()
        else:
            answer = raw_answer.strip()

        logger.info(
            "Groq generated answer (%s): %d input tokens, %d output tokens, %.0fms",
            self.model_name,
            input_tokens,
            output_tokens,
            latency_ms,
        )

        return GenerationResult(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model_name=f"groq/{self.model_name}",
            prompt=prompt,
            metadata=metadata,
        )

    def _generate_hf(self, prompt: str) -> GenerationResult:
        """Generate an answer using local HuggingFace pipeline."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_tokens = input_ids.shape[1]

        start = time.time()

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self._task == "text-generation":
            gen_kwargs["return_full_text"] = False

        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        output = self.pipeline(prompt, **gen_kwargs)

        latency_ms = (time.time() - start) * 1000

        # Extract answer text
        if isinstance(output, list) and len(output) > 0:
            raw_answer = output[0].get("generated_text", "").strip()
        else:
            raw_answer = str(output).strip()

        # If pipeline prepended prompt, strip it
        if raw_answer.startswith(prompt):
            raw_answer = raw_answer[len(prompt):].strip()

        # Remove trailing stop tokens if any
        for stop_token in ["<|im_end|>", "<|endoftext|>"]:
            if raw_answer.endswith(stop_token):
                raw_answer = raw_answer[:-len(stop_token)].strip()

        metadata = {}
        # Parse reasoning tags if model produced thinking blocks (<think>...</think>)
        if "<think>" in raw_answer and "</think>" in raw_answer:
            think_start = raw_answer.find("<think>") + len("<think>")
            think_end = raw_answer.find("</think>")
            metadata["reasoning"] = raw_answer[think_start:think_end].strip()
            final_answer = raw_answer[think_end + len("</think>"):].strip()
            metadata["reasoning"] = metadata["reasoning"]
            answer = final_answer if final_answer else raw_answer
        else:
            answer = raw_answer

        output_ids = self.tokenizer.encode(answer)
        output_tokens = len(output_ids)

        result = GenerationResult(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model_name=self.model_name,
            prompt=prompt,
            metadata=metadata,
        )

        logger.info(
            "Generated answer: %d input tokens, %d output tokens, %.0fms",
            input_tokens,
            output_tokens,
            latency_ms,
        )
        return result

    def generate(self, prompt: str) -> GenerationResult:
        """
        Generate an answer from a prompt.

        Args:
            prompt: Complete prompt (system + context + question).

        Returns:
            GenerationResult with answer, token counts, latency, and reasoning metadata.
        """
        prompt = self._truncate_prompt(prompt)
        if self.is_groq:
            return self._generate_groq(prompt)
        return self._generate_hf(prompt)
