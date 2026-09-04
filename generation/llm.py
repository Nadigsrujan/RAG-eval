"""
LLM Generator — wraps HuggingFace models for RAG answer generation.

Supports FLAN-T5 (encoder-decoder) and other text-generation models.
Records token counts and latency for observability.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Known model aliases for convenience
MODEL_ALIASES = {
    "qwen3-4b-instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen/qwen3-4b-instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "flan-t5": "google/flan-t5-base",
    "flan-t5-base": "google/flan-t5-base",
    "google/flan-t5-base": "google/flan-t5-base",
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

    Uses HuggingFace transformers pipeline with lazy loading.
    Supports causal language models (e.g., Qwen3, Mistral, Llama)
    and encoder-decoder models (e.g., FLAN-T5).

    Args:
        model_name: HuggingFace model identifier or alias.
        max_new_tokens: Maximum tokens to generate.
        max_input_tokens: Maximum input tokens (truncates longer prompts).
        temperature: Sampling temperature (0 = greedy).
        device: Device to run on. None = auto-detect (mps, cuda, cpu).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        max_new_tokens: int = 500,
        max_input_tokens: int = 2048,
        temperature: float = 0.1,
        device: Optional[str] = None,
    ):
        # Resolve aliases (case-insensitive)
        clean_name = model_name.strip()
        self.model_name = MODEL_ALIASES.get(clean_name.lower(), clean_name)
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.temperature = temperature
        self._pipeline = None
        self._tokenizer = None
        self._device = device
        self._task = None

    @property
    def pipeline(self):
        """Lazy-load the generation pipeline."""
        if self._pipeline is None:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
                pipeline,
            )

            logger.info("Loading generation model: %s", self.model_name)

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
        if self._tokenizer is None:
            _ = self.pipeline  # triggers lazy load
        return self._tokenizer

    def _truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt to max_input_tokens, preserving head and tail."""
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

    def generate(self, prompt: str) -> GenerationResult:
        """
        Generate an answer from a prompt.

        Args:
            prompt: Complete prompt (system + context + question).

        Returns:
            GenerationResult with answer, token counts, latency, and reasoning metadata.
        """
        prompt = self._truncate_prompt(prompt)

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
            reasoning = raw_answer[think_start:think_end].strip()
            final_answer = raw_answer[think_end + len("</think>"):].strip()
            metadata["reasoning"] = reasoning
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

