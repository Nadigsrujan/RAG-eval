"""
LLM-as-Judge — uses a separate LLM to evaluate answer quality.

Evaluates:
  - Correctness: Does the answer match the ground truth?
  - Faithfulness: Is every claim grounded in the evidence?
  - Relevance: Does the answer address the question?
  - Citation Quality: Are citations accurate and complete?

Outputs structured JSON scores with reasoning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Judge prompt template
JUDGE_PROMPT = """You are an expert evaluator for a document question-answering system.

Given the following information, score the answer on each dimension from 1 to 5.

QUESTION: {question}

RETRIEVED EVIDENCE:
{evidence}

GENERATED ANSWER:
{answer}

GROUND TRUTH ANSWER:
{ground_truth}

Score each dimension from 1 (worst) to 5 (best):

1. **Correctness** (1-5): Does the answer agree with the ground truth?
   - 1: Completely wrong
   - 3: Partially correct
   - 5: Fully correct

2. **Faithfulness** (1-5): Is every claim in the answer supported by the evidence?
   - 1: Major hallucinations
   - 3: Some unsupported claims
   - 5: Fully grounded in evidence

3. **Relevance** (1-5): Does the answer address the question?
   - 1: Off-topic
   - 3: Partially relevant
   - 5: Directly and fully addresses the question

4. **Citation Quality** (1-5): Are source citations accurate and helpful?
   - 1: No citations or all wrong
   - 3: Some correct citations
   - 5: All claims properly cited

Respond ONLY with valid JSON in this exact format:
{{
    "correctness": <int>,
    "faithfulness": <int>,
    "relevance": <int>,
    "citation_quality": <int>,
    "reasoning": "<brief explanation>"
}}"""


@dataclass
class JudgeScores:
    """Scores from the LLM judge."""

    correctness: int = 0
    faithfulness: int = 0
    relevance: int = 0
    citation_quality: int = 0
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "correctness": self.correctness,
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "citation_quality": self.citation_quality,
            "reasoning": self.reasoning,
        }

    @property
    def average_score(self) -> float:
        scores = [self.correctness, self.faithfulness, self.relevance, self.citation_quality]
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class JudgeMetrics:
    """Aggregated judge evaluation metrics."""

    avg_correctness: float = 0.0
    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    avg_citation_quality: float = 0.0
    avg_overall: float = 0.0
    num_queries: int = 0
    individual_scores: list[JudgeScores] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "avg_correctness": round(self.avg_correctness, 2),
            "avg_faithfulness": round(self.avg_faithfulness, 2),
            "avg_relevance": round(self.avg_relevance, 2),
            "avg_citation_quality": round(self.avg_citation_quality, 2),
            "avg_overall": round(self.avg_overall, 2),
            "num_queries": self.num_queries,
        }


class LLMJudge:
    """
    Uses an LLM to evaluate RAG answer quality with structured scoring.

    The judge model should ideally be different from (or larger than) the
    generation model to avoid self-evaluation bias.

    Args:
        generator: A Generator instance to use as the judge.
    """

    def __init__(self, generator=None):
        self._generator = generator

    @property
    def generator(self):
        if self._generator is None:
            from generation.llm import Generator
            self._generator = Generator(
                model_name="Qwen/Qwen3-4B-Instruct-2507",
                max_new_tokens=300,
                temperature=0.0,
            )
        return self._generator

    def judge_single(
        self,
        question: str,
        evidence: str,
        answer: str,
        ground_truth: str,
    ) -> JudgeScores:
        """
        Score a single question-answer pair.

        Args:
            question: The original question.
            evidence: The retrieved evidence used to generate the answer.
            answer: The generated answer.
            ground_truth: The expected correct answer.

        Returns:
            JudgeScores with individual dimension scores.
        """
        prompt = JUDGE_PROMPT.format(
            question=question,
            evidence=evidence[:1500],  # truncate to avoid context overflow
            answer=answer,
            ground_truth=ground_truth,
        )

        try:
            result = self.generator.generate(prompt)
            scores = self._parse_scores(result.answer)
            return scores
        except Exception as e:
            logger.warning("LLM judge failed for question '%s': %s", question[:50], e)
            return JudgeScores(reasoning=f"Judge error: {e}")

    def evaluate(
        self,
        eval_data: list[dict],
        pipeline_fn,
    ) -> JudgeMetrics:
        """
        Run LLM judge evaluation across all queries.

        Args:
            eval_data: List of eval items with question, answer, relevant_chunks.
            pipeline_fn: Callable(question) -> {"answer": str, "evidence": str}

        Returns:
            Aggregated JudgeMetrics.
        """
        all_scores: list[JudgeScores] = []

        for item in eval_data:
            question = item["question"]
            ground_truth = item.get("answer", "")

            result = pipeline_fn(question)
            answer = result.get("answer", "")
            evidence = result.get("evidence", "")

            scores = self.judge_single(question, evidence, answer, ground_truth)
            all_scores.append(scores)

        n = len(all_scores)
        if n == 0:
            return JudgeMetrics()

        metrics = JudgeMetrics(
            avg_correctness=sum(s.correctness for s in all_scores) / n,
            avg_faithfulness=sum(s.faithfulness for s in all_scores) / n,
            avg_relevance=sum(s.relevance for s in all_scores) / n,
            avg_citation_quality=sum(s.citation_quality for s in all_scores) / n,
            avg_overall=sum(s.average_score for s in all_scores) / n,
            num_queries=n,
            individual_scores=all_scores,
        )

        logger.info(
            "LLM Judge eval: %d queries — Correctness=%.1f, Faithfulness=%.1f, Overall=%.1f",
            n,
            metrics.avg_correctness,
            metrics.avg_faithfulness,
            metrics.avg_overall,
        )
        return metrics

    @staticmethod
    def _parse_scores(response: str) -> JudgeScores:
        """Parse structured JSON scores from judge response."""
        try:
            # Try to extract JSON from the response
            # Handle cases where the model wraps JSON in markdown
            json_str = response.strip()
            if "```" in json_str:
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]

            data = json.loads(json_str)
            return JudgeScores(
                correctness=int(data.get("correctness", 0)),
                faithfulness=int(data.get("faithfulness", 0)),
                relevance=int(data.get("relevance", 0)),
                citation_quality=int(data.get("citation_quality", 0)),
                reasoning=str(data.get("reasoning", "")),
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Failed to parse judge response: %s", e)
            return JudgeScores(reasoning=f"Parse error: {response[:200]}")
