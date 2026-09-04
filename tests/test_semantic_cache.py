"""Tests for SemanticCache component and API caching integration."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from guardrails.semantic_cache import SemanticCache

# ── Unit Tests ─────────────────────────────────────────────────


class TestSemanticCacheUnit:
    """Unit tests for SemanticCache logic and boundary conditions."""

    def test_cache_empty(self):
        """Empty cache returns None."""
        cache = SemanticCache(similarity_threshold=0.95)
        assert len(cache) == 0
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.array([1.0, 0.0, 0.0])
        assert cache.check("Any query?", embedder=mock_embedder) is None

    def test_cache_store_and_retrieve(self):
        """Store then check retrieves correctly."""
        cache = SemanticCache(similarity_threshold=0.95)
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result_payload = {"answer": "Stored answer", "metadata": {"test": True}}

        cache.store("What is X?", vec, result_payload)
        assert len(cache) == 1

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = vec

        retrieved = cache.check("What is X?", embedder=mock_embedder)
        assert retrieved is not None
        assert retrieved["answer"] == "Stored answer"
        assert retrieved["metadata"]["cached"] is True
        assert retrieved["metadata"]["cached_similarity"] >= 0.99

    def test_cache_check_exact_match(self):
        """Identical queries should match with similarity approx 1.0."""
        cache = SemanticCache(similarity_threshold=0.95)
        vec = np.array([0.6, 0.8, 0.0], dtype=np.float32)  # unit vector: 0.6^2 + 0.8^2 = 1.0
        cache.store("What is revenue?", vec, {"answer": "Revenue is $10M", "metadata": {}})

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = vec.copy()

        hit = cache.check("What is revenue?", embedder=mock_embedder)
        assert hit is not None
        assert hit["answer"] == "Revenue is $10M"
        assert pytest.approx(hit["metadata"]["cached_similarity"], abs=0.001) == 1.0

    def test_cache_check_slight_variation(self):
        """Slightly rephrased queries should match with similarity >= 0.95."""
        cache = SemanticCache(similarity_threshold=0.95)
        # Angle theta where cos(theta) = 0.96
        # v1 = [1, 0], v2 = [cos(theta), sin(theta)] = [0.96, sqrt(1 - 0.96^2)] = [0.96, 0.28]
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.96, 0.28], dtype=np.float32)

        cache.store("What is the company profit?", v1, {"answer": "$2M", "metadata": {}})

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = v2

        hit = cache.check("Can you tell me the profit of the company?", embedder=mock_embedder)
        assert hit is not None
        assert hit["answer"] == "$2M"
        assert hit["metadata"]["cached_similarity"] >= 0.95

    def test_cache_check_no_match(self):
        """Dissimilar queries should not match cache."""
        cache = SemanticCache(similarity_threshold=0.95)
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)  # orthogonal, similarity = 0.0

        cache.store("What is the CEO name?", v1, {"answer": "Alice", "metadata": {}})

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = v2

        hit = cache.check("How does the battery work?", embedder=mock_embedder)
        assert hit is None

    def test_cache_threshold_boundary(self):
        """Queries at exactly 0.95 vs 0.94 threshold boundary."""
        cache = SemanticCache(similarity_threshold=0.95)
        base_v = np.array([1.0, 0.0], dtype=np.float32)
        cache.store("Base query", base_v, {"answer": "Base answer", "metadata": {}})

        mock_embedder = MagicMock()

        # Case 1: sim = 0.95 (at boundary) -> should hit
        v_95 = np.array([0.95, np.sqrt(1.0 - 0.95**2)], dtype=np.float32)
        mock_embedder.embed_query.return_value = v_95
        assert cache.check("Boundary query at 0.95", embedder=mock_embedder) is not None

        # Case 2: sim = 0.94 (below boundary) -> should miss
        v_94 = np.array([0.94, np.sqrt(1.0 - 0.94**2)], dtype=np.float32)
        mock_embedder.embed_query.return_value = v_94
        assert cache.check("Boundary query at 0.94", embedder=mock_embedder) is None

    def test_cache_clear(self):
        """Clear empties all stored items."""
        cache = SemanticCache(similarity_threshold=0.95)
        cache.store("Q1", np.array([1.0, 0.0]), {"answer": "A1", "metadata": {}})
        cache.store("Q2", np.array([0.0, 1.0]), {"answer": "A2", "metadata": {}})
        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.array([1.0, 0.0])
        assert cache.check("Q1", embedder=mock_embedder) is None


# ── Integration Tests ──────────────────────────────────────────


class TestSemanticCacheAPIIntegration:
    """Integration tests testing API endpoints with semantic cache."""

    def test_query_endpoint_cache_hit(self):
        """API /query returns cached result on similar question."""
        import api.main as main_mod
        from api.main import app

        client = TestClient(app)

        # Mock global components
        mock_retrieval = MagicMock()
        mock_gen = MagicMock()
        mock_embedder = MagicMock()

        # Configure embedder with fixed embeddings
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.97, 0.24, 0.0], dtype=np.float32)  # similarity ~ 0.97 >= 0.95
        v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)   # dissimilar

        def fake_embed(q):
            if "first" in q:
                return v1
            elif "similar" in q:
                return v2
            return v3

        mock_embedder.embed_query.side_effect = fake_embed

        old_retrieval = main_mod._retrieval_pipeline
        old_gen = main_mod._generator
        old_embedder = main_mod._embedder
        old_orch = main_mod._orchestrator
        old_prompt = main_mod._prompt_manager

        mock_prompt = MagicMock()
        mock_prompt.format.return_value = "formatted prompt"

        try:
            main_mod._retrieval_pipeline = mock_retrieval
            main_mod._generator = mock_gen
            main_mod._embedder = mock_embedder
            main_mod._orchestrator = None  # test direct pipeline flow
            main_mod._prompt_manager = mock_prompt
            main_mod._semantic_cache.clear()

            # Mock pipeline retrieve & generate
            mock_retrieval.retrieve.return_value = MagicMock(chunks=[], timings={"retrieval_ms": 10.0, "reranking_ms": 5.0}, metadata={})
            mock_gen.generate.return_value = MagicMock(answer="First answer", input_tokens=10, output_tokens=5, model_name="test-model")

            # 1. Send first query -> triggers pipeline & caches
            resp1 = client.post("/query", json={"question": "first query", "use_orchestrator": False})
            assert resp1.status_code == 200
            data1 = resp1.json()
            assert data1["answer"] == "First answer"
            assert data1["metadata"].get("cached") is not True
            assert len(main_mod._semantic_cache) == 1
            assert mock_retrieval.retrieve.call_count == 1

            # 2. Send similar query -> should return cached result without calling retrieve
            resp2 = client.post("/query", json={"question": "similar query", "use_orchestrator": False})
            assert resp2.status_code == 200
            data2 = resp2.json()
            assert data2["answer"] == "First answer"
            assert data2["metadata"].get("cached") is True
            assert data2["metadata"].get("cached_similarity") >= 0.95
            # Retrieval pipeline should NOT have been called again
            assert mock_retrieval.retrieve.call_count == 1
            # Timings should show 0ms for retrieval/generation
            assert data2["timings"]["retrieval_ms"] == 0.0
            assert data2["timings"]["generation_ms"] == 0.0

            # 3. Send unrelated query -> should call pipeline again
            resp3 = client.post("/query", json={"question": "unrelated query", "use_orchestrator": False})
            assert resp3.status_code == 200
            assert mock_retrieval.retrieve.call_count == 2
            assert len(main_mod._semantic_cache) == 2

        finally:
            main_mod._retrieval_pipeline = old_retrieval
            main_mod._generator = old_gen
            main_mod._embedder = old_embedder
            main_mod._orchestrator = old_orch
            main_mod._prompt_manager = old_prompt
            main_mod._semantic_cache.clear()

    def test_orchestrate_query_cache_hit(self):
        """API /orchestrate/query returns cached result on similar question."""
        import api.main as main_mod
        from api.main import app

        client = TestClient(app)

        mock_orch = MagicMock()
        mock_retrieval = MagicMock()
        mock_embedder = MagicMock()

        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.98, 0.19], dtype=np.float32)  # similarity ~ 0.98 >= 0.95

        def fake_embed(q):
            if "first" in q:
                return v1
            return v2

        mock_embedder.embed_query.side_effect = fake_embed

        mock_state = MagicMock()
        mock_state.final_answer = "Orchestrated answer"
        mock_state.intent = "direct"
        mock_state.sub_queries = []
        mock_state.sources = []
        mock_state.citations = []
        mock_state.thoughts = []
        mock_state.retrieval_valid = True
        mock_state.retrieval_score = 0.9
        mock_state.faithfulness_score = 1.0
        mock_state.citation_accuracy = 1.0
        mock_state.self_corrected = False
        mock_state.validation_reasons = []
        mock_state.timings = {"retrieval_agent_ms": 15.0, "generation_agent_ms": 25.0, "reflection_agent_ms": 5.0, "total_ms": 45.0}
        mock_state.retrieval_diagnostics = {"internal_timings": {"reranking_ms": 5.0}}
        mock_state.metadata = {}

        mock_orch.orchestrate_query.return_value = mock_state

        old_orch = main_mod._orchestrator
        old_retrieval = main_mod._retrieval_pipeline
        old_embedder = main_mod._embedder

        try:
            main_mod._orchestrator = mock_orch
            main_mod._retrieval_pipeline = mock_retrieval
            main_mod._embedder = mock_embedder
            main_mod._semantic_cache.clear()

            # 1. First orchestrated query
            r1 = client.post("/orchestrate/query", json={"question": "first question"})
            assert r1.status_code == 200
            d1 = r1.json()
            assert d1["answer"] == "Orchestrated answer"
            assert mock_orch.orchestrate_query.call_count == 1

            # 2. Rephrased query -> cache hit
            r2 = client.post("/orchestrate/query", json={"question": "similar question"})
            assert r2.status_code == 200
            d2 = r2.json()
            assert d2["answer"] == "Orchestrated answer"
            assert d2["metadata"].get("cached") is True
            assert mock_orch.orchestrate_query.call_count == 1  # Not called again
            assert d2["timings"]["retrieval_ms"] == 0.0
            assert d2["timings"]["generation_ms"] == 0.0

        finally:
            main_mod._orchestrator = old_orch
            main_mod._retrieval_pipeline = old_retrieval
            main_mod._embedder = old_embedder
            main_mod._semantic_cache.clear()

