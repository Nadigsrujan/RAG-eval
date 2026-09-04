"""Tests for the evaluation framework — retrieval, generation, citation metrics."""


import pytest

from evaluation.citation_eval import CitationEvaluator
from evaluation.generation_eval import GenerationEvaluator
from evaluation.retrieval_eval import RetrievalEvaluator

# ── Retrieval Evaluator Tests ──────────────────────────────────


class TestRetrievalEvaluator:
    eval = RetrievalEvaluator()

    def test_recall_at_k_perfect(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert self.eval.recall_at_k(retrieved, relevant, 3) == 1.0

    def test_recall_at_k_partial(self):
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b"}
        assert self.eval.recall_at_k(retrieved, relevant, 3) == 0.5

    def test_recall_at_k_zero(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert self.eval.recall_at_k(retrieved, relevant, 3) == 0.0

    def test_mrr_first_position(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert self.eval.reciprocal_rank(retrieved, relevant) == 1.0

    def test_mrr_second_position(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a"}
        assert self.eval.reciprocal_rank(retrieved, relevant) == 0.5

    def test_mrr_not_found(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert self.eval.reciprocal_rank(retrieved, relevant) == 0.0

    def test_ndcg_perfect_ranking(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        ndcg = self.eval.ndcg_at_k(retrieved, relevant, 3)
        assert ndcg == pytest.approx(1.0)

    def test_ndcg_imperfect_ranking(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a", "b"}
        ndcg = self.eval.ndcg_at_k(retrieved, relevant, 3)
        assert 0 < ndcg < 1.0

    def test_precision_at_k(self):
        retrieved = ["a", "x", "b", "y", "z"]
        relevant = {"a", "b"}
        assert self.eval.precision_at_k(retrieved, relevant, 5) == pytest.approx(0.4)


# ── Generation Evaluator Tests ─────────────────────────────────


class TestGenerationEvaluator:
    eval = GenerationEvaluator()

    def test_exact_match_true(self):
        assert self.eval.exact_match("AdamW", "adamw") is True

    def test_exact_match_false(self):
        assert self.eval.exact_match("SGD", "AdamW") is False

    def test_contains_match(self):
        assert self.eval.contains_match("The optimizer is AdamW with lr 3e-4", "adamw") is True

    def test_token_f1_perfect(self):
        f1 = self.eval.token_f1("the cat sat", "the cat sat")
        assert f1 == pytest.approx(1.0)

    def test_token_f1_partial(self):
        f1 = self.eval.token_f1("the cat sat on the mat", "the cat")
        assert 0 < f1 < 1.0

    def test_token_f1_no_overlap(self):
        f1 = self.eval.token_f1("hello world", "foo bar")
        assert f1 == pytest.approx(0.0)

    def test_normalize(self):
        result = self.eval.normalize("  Hello, World!  ")
        assert result == "hello world"


# ── Citation Evaluator Tests ───────────────────────────────────


class TestCitationEvaluator:
    eval = CitationEvaluator()

    def test_perfect_citations(self):
        generated = {("paper.pdf", 4), ("paper.pdf", 5)}
        expected = {("paper.pdf", 4), ("paper.pdf", 5)}
        assert self.eval.citation_precision(generated, expected) == 1.0
        assert self.eval.citation_recall(generated, expected) == 1.0

    def test_partial_citations(self):
        generated = {("paper.pdf", 4), ("paper.pdf", 99)}
        expected = {("paper.pdf", 4), ("paper.pdf", 5)}
        assert self.eval.citation_precision(generated, expected) == 0.5
        assert self.eval.citation_recall(generated, expected) == 0.5

    def test_no_citations(self):
        generated = set()
        expected = {("paper.pdf", 4)}
        assert self.eval.citation_precision(generated, expected) == 0.0
        assert self.eval.citation_recall(generated, expected) == 0.0

    def test_f1_computation(self):
        f1 = self.eval.citation_f1(0.5, 0.5)
        assert f1 == pytest.approx(0.5)

        f1 = self.eval.citation_f1(1.0, 1.0)
        assert f1 == pytest.approx(1.0)

        f1 = self.eval.citation_f1(0.0, 0.0)
        assert f1 == pytest.approx(0.0)
