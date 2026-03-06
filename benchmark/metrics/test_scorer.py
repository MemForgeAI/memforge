"""Tests for the scoring engine."""

from benchmark.metrics.scorer import (
    score_string_match,
    compute_precision_at_k,
    compute_mrr,
    compute_noise_rejection,
    compute_token_efficiency,
    compute_recall_at_k,
)
from benchmark.models import ExpectedMemory


class TestStringMatchScoring:
    def test_exact_match(self) -> None:
        memories = ["Alex switched to Cursor last week"]
        expected = [ExpectedMemory(content_match="Cursor", required=True, rank=1)]
        hits = score_string_match(memories, expected)
        assert len(hits) == 1
        assert hits[0] is True

    def test_no_match(self) -> None:
        memories = ["Alex uses TypeScript"]
        expected = [ExpectedMemory(content_match="Python", required=True)]
        hits = score_string_match(memories, expected)
        assert hits[0] is False

    def test_case_insensitive(self) -> None:
        memories = ["alex prefers CURSOR editor"]
        expected = [ExpectedMemory(content_match="cursor", required=True)]
        hits = score_string_match(memories, expected)
        assert hits[0] is True

    def test_multiple_expected(self) -> None:
        memories = ["Alex uses Cursor", "Alex likes TypeScript"]
        expected = [
            ExpectedMemory(content_match="Cursor", required=True),
            ExpectedMemory(content_match="Python", required=True),
        ]
        hits = score_string_match(memories, expected)
        assert hits[0] is True
        assert hits[1] is False


class TestPrecisionAtK:
    def test_perfect_precision(self) -> None:
        relevant = [True, True, True, True, True]
        assert compute_precision_at_k(relevant, k=5) == 1.0

    def test_half_precision(self) -> None:
        relevant = [True, False, True, False, False]
        assert compute_precision_at_k(relevant, k=5) == 0.4

    def test_k_larger_than_results(self) -> None:
        relevant = [True, True]
        assert compute_precision_at_k(relevant, k=5) == 0.4

    def test_zero_k(self) -> None:
        assert compute_precision_at_k([], k=0) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        assert compute_recall_at_k([True, True], 2, 5) == 1.0

    def test_partial_recall(self) -> None:
        assert compute_recall_at_k([True, False], 2, 5) == 0.5

    def test_zero_relevant(self) -> None:
        assert compute_recall_at_k([], 0, 5) == 0.0


class TestMRR:
    def test_first_hit(self) -> None:
        relevant = [True, False, False]
        assert compute_mrr(relevant) == 1.0

    def test_second_hit(self) -> None:
        relevant = [False, True, False]
        assert compute_mrr(relevant) == 0.5

    def test_no_hits(self) -> None:
        relevant = [False, False, False]
        assert compute_mrr(relevant) == 0.0


class TestNoiseRejection:
    def test_all_noise_rejected(self) -> None:
        memories = ["Alex uses TypeScript"]
        noise_terms = ["weather", "lunch", "restaurant"]
        assert compute_noise_rejection(memories, noise_terms) == 1.0

    def test_some_noise_present(self) -> None:
        memories = ["Alex uses TypeScript", "Had lunch at the restaurant"]
        noise_terms = ["lunch", "restaurant"]
        assert compute_noise_rejection(memories, noise_terms) < 1.0

    def test_no_noise_terms(self) -> None:
        assert compute_noise_rejection(["anything"], []) == 1.0


class TestTokenEfficiency:
    def test_all_relevant(self) -> None:
        assert compute_token_efficiency(100, 100) == 1.0

    def test_half_relevant(self) -> None:
        assert compute_token_efficiency(50, 100) == 0.5

    def test_zero_tokens(self) -> None:
        assert compute_token_efficiency(0, 0) == 0.0
