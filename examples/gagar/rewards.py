# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keep binary outcomes alongside equivalent GAGAR training rewards."""

from examples.gagar.gagar import compute_group_rewards


def reward_records(
    original: list[float], corrected: list[float], weights: list[float], *, fallback: bool = False
) -> list[dict]:
    scores = compute_group_rewards(corrected, weights)
    return [
        {
            "score": score,
            "test_reward": test,
            "outcome_reward": reward,
            "quality_weight": weight,
            "grader_fallback": int(fallback),
        }
        for score, test, reward, weight in zip(scores, original, corrected, weights, strict=True)
    ]
