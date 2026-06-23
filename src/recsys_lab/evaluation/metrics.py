import numpy as np


def recall_at_k(
    recommended_items: list[int],
    relevant_items: set[int],
    k: int,
) -> float:
    """Compute Recall@K.

    Recall@K = relevant recommended items in top K / total relevant items.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    if not relevant_items:
        return 0.0

    top_k = recommended_items[:k]
    hits = sum(item in relevant_items for item in top_k)

    return hits / len(relevant_items)


def precision_at_k(
    recommended_items: list[int],
    relevant_items: set[int],
    k: int,
) -> float:
    """Compute Precision@K.

    Precision@K = relevant recommended items in top K / K.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    top_k = recommended_items[:k]

    if not top_k:
        return 0.0

    hits = sum(item in relevant_items for item in top_k)

    return hits / k


def dcg_at_k(
    recommended_items: list[int],
    relevant_items: set[int],
    k: int,
) -> float:
    """Compute Discounted Cumulative Gain at K for binary relevance."""
    if k <= 0:
        raise ValueError("k must be positive.")

    top_k = recommended_items[:k]
    score = 0.0

    # Recommendations decrease in importance as further down the recommended_items list,
    # So, Ideally, the items in relevant list should be recommended 

    for index, item in enumerate(top_k):
        if item in relevant_items:
            rank = index + 1
            score += 1.0 / np.log2(rank + 1)

    return score


def ndcg_at_k(
    recommended_items: list[int],
    relevant_items: set[int],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at K."""
    if k <= 0:
        raise ValueError("k must be positive.")

    if not relevant_items:
        return 0.0

    dcg = dcg_at_k(recommended_items, relevant_items, k)

    #ideally everything that is recommended is on the relevant list

    ideal_hits = min(len(relevant_items), k)
    ideal_dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def average_precision_at_k(
    recommended_items: list[int],
    relevant_items: set[int],
    k: int,
) -> float:
    """Compute Average Precision@K.

    This rewards putting relevant items early in the ranking.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    if not relevant_items:
        return 0.0

    top_k = recommended_items[:k]
    precisions = []

    for index, item in enumerate(top_k):
        if item in relevant_items:
            rank = index + 1
            precision_at_rank = precision_at_k(
                recommended_items=recommended_items,
                relevant_items=relevant_items,
                k=rank,
            )
            precisions.append(precision_at_rank)

    if not precisions:
        return 0.0

    return sum(precisions) / min(len(relevant_items), k)