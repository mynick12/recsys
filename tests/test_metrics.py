import pytest

from recsys_lab.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k_perfect():
    recommended = [1, 2, 3]
    relevant = {1, 2}

    assert recall_at_k(recommended, relevant, k=2) == 1.0


def test_recall_at_k_partial():
    recommended = [1, 3, 4]
    relevant = {1, 2}

    assert recall_at_k(recommended, relevant, k=2) == 0.5


def test_precision_at_k():
    recommended = [1, 3, 4]
    relevant = {1, 2}

    assert precision_at_k(recommended, relevant, k=2) == 0.5


def test_ndcg_at_k_perfect_is_one():
    recommended = [1, 2, 3]
    relevant = {1, 2}

    assert ndcg_at_k(recommended, relevant, k=2) == pytest.approx(1.0)


def test_ndcg_at_k_worse_ranking_is_lower():
    good_recommendation = [1, 2, 3]
    bad_recommendation = [3, 1, 2]
    relevant = {1, 2}

    assert ndcg_at_k(good_recommendation, relevant, k=3) > ndcg_at_k(
        bad_recommendation,
        relevant,
        k=3,
    )


def test_average_precision_at_k():
    recommended = [1, 3, 2]
    relevant = {1, 2}

    score = average_precision_at_k(recommended, relevant, k=3)

    assert score == pytest.approx((1.0 + 2 / 3) / 2)


def test_empty_relevant_items():
    recommended = [1, 2, 3]
    relevant = set()

    assert recall_at_k(recommended, relevant, k=3) == 0.0
    assert ndcg_at_k(recommended, relevant, k=3) == 0.0
    assert average_precision_at_k(recommended, relevant, k=3) == 0.0


def test_invalid_k_raises_error():
    recommended = [1, 2, 3]
    relevant = {1}

    with pytest.raises(ValueError):
        recall_at_k(recommended, relevant, k=0)