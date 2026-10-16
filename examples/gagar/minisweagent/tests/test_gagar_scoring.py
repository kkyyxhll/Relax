# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy

import pytest

from examples.gagar.minisweagent.scoring import CRITERIA, grade_to_weights, quality_tier


def evaluation(index, scores, **kwargs):
    return {
        "rollout_n": index,
        "scores": dict(zip(CRITERIA, scores, strict=True)),
        "issues": [],
        "evidence": "task-specific diff and test evidence",
        **kwargs,
    }


def test_paper_tier_mapping_and_tied_group_ranks():
    evaluations = [
        evaluation(0, [5] * 5),
        evaluation(1, [4] * 5),
        evaluation(2, [3] * 5),
        evaluation(3, [3] * 5),
        evaluation(4, [2, 3, 3, 3, 3]),
        evaluation(5, [1, 4, 4, 4, 4]),
    ]
    result = grade_to_weights([1] * 6 + [0], {"evaluations": evaluations, "ranking": [[0], [1], [2, 3], [4], [5]]})
    assert result["weights"] == pytest.approx([1, 0.9, 0.85, 0.85, 0.4, 0.2, 0])
    assert result["rewards"] == [1] * 6 + [0]


def test_t2_single_tied_group_gets_maximum_weight():
    result = grade_to_weights(
        [1, 1, 0], {"evaluations": [evaluation(0, [3] * 5), evaluation(1, [3] * 5)], "ranking": [[0, 1]]}
    )
    assert result["weights"] == [0.85, 0.85, 0]


@pytest.mark.parametrize(
    "scores, issues, expected",
    [
        ([5] * 5, ["unrequested_rewrite"], 3),
        ([5] * 5, ["test_specific_workaround"], 3),
        ([5, 5, 2, 2, 5], [], 3),
        ([1, 5, 5, 5, 5], [], 3),
        ([5] * 5, ["severe_process_issue"], 2),
        ([5] * 5, ["unresolved_regression"], 2),
        ([4] * 5, [], 1),
        ([4, 4, 4, 4, 3], [], 2),
    ],
)
def test_tier_rules_apply_in_paper_order(scores, issues, expected):
    assert quality_tier(dict(zip(CRITERIA, scores, strict=True)), issues) == expected


def test_confirmed_external_solution_changes_outcome_before_mapping():
    original = [1, 1, 0]
    grading = {
        "evaluations": [
            evaluation(0, [5] * 5),
            evaluation(
                1,
                [5] * 5,
                confirmed_external_solution=True,
                hack_evidence="original trajectory downloaded leaked solution",
            ),
        ],
        "ranking": [[0]],
    }
    result = grade_to_weights(original, grading)
    assert result["rewards"] == [1, 0, 0] and original == [1, 1, 0]
    assert result["weights"] == [1, 0, 0]
    assert result["evaluations"][1]["tier"] is None


def test_test_workaround_is_quality_issue_not_automatic_outcome_correction():
    result = grade_to_weights(
        [1, 0], {"evaluations": [evaluation(0, [5] * 5, issues=["test_specific_workaround"])], "ranking": [[0]]}
    )
    assert result["rewards"] == [1, 0] and result["weights"] == [0.2, 0]


def test_reordering_requires_evidence_for_each_changed_candidate():
    grading = {"evaluations": [evaluation(0, [5] * 5), evaluation(1, [4] * 5)], "ranking": [[1], [0]]}
    with pytest.raises(ValueError, match="rank_reason"):
        grade_to_weights([1, 1, 0], grading)
    for entry in grading["evaluations"]:
        entry["rank_reason"] = "repository-specific evidence justifies reversal"
    assert grade_to_weights([1, 1, 0], grading)["weights"] == [0.9, 1, 0]


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "wrong_tier", "bad_score", "hack_without_evidence", "rank_failure"]
)
def test_invalid_model_assessments_are_rejected(mutation):
    grading = {"evaluations": [evaluation(0, [5] * 5), evaluation(1, [3] * 5)], "ranking": [[0], [1]]}
    if mutation == "duplicate":
        grading["evaluations"].append(copy.deepcopy(grading["evaluations"][0]))
    elif mutation == "missing":
        grading["ranking"] = [[0]]
    elif mutation == "wrong_tier":
        grading["ranking"] = [[1], [0]]
    elif mutation == "bad_score":
        grading["evaluations"][0]["scores"]["approach"] = 0
    elif mutation == "hack_without_evidence":
        grading["evaluations"][0]["confirmed_external_solution"] = True
    else:
        grading["ranking"] = [[0], [1], [2]]
    with pytest.raises(ValueError):
        grade_to_weights([1, 1, 0], grading)


def test_ranking_error_identifies_original_rollout_id():
    grading = {"evaluations": [evaluation(6, [5] * 5)], "ranking": [[0]]}
    with pytest.raises(ValueError, match=r"expected \[6\], got 0"):
        grade_to_weights([0, 0, 0, 0, 0, 0, 1, 0], grading)


def test_missing_ranking_identifies_omitted_rollout_ids():
    grading = {"evaluations": [evaluation(1, [5] * 5), evaluation(6, [5] * 5)], "ranking": [[1]]}
    with pytest.raises(ValueError, match=r"missing corrected passing candidates: \[6\]"):
        grade_to_weights([0, 1, 0, 0, 0, 0, 1, 0], grading)


@pytest.mark.parametrize("ranking", [[[0, 3], [5, 6]], [[3], [0, 5, 6]]])
@pytest.mark.parametrize("correct_external", [False, True])
def test_tier_rejection_identifies_corrected_candidates_and_allows_repair(ranking, correct_external):
    evaluations = [evaluation(i, [4] * 5) for i in (0, 5, 6)] + [evaluation(3, [4, 4, 3, 4, 3])]
    if correct_external:
        evaluations.append(
            evaluation(7, [5] * 5, confirmed_external_solution=True, hack_evidence="trajectory used leaked solution")
        )
    rewards = [1, 0, 0, 1, 0, 1, 1, int(correct_external)]
    grading = {"evaluations": evaluations, "ranking": ranking}
    with pytest.raises(ValueError) as exc:
        grade_to_weights(rewards, grading)
    assert "candidate IDs by tier: {'T1': [0, 5, 6], 'T2': [3], 'T3': []}" in str(exc.value)
    grading["ranking"] = [[0, 5, 6], [3]]
    result = grade_to_weights(rewards, grading)
    assert result["weights"] == [1, 0, 0, 0.85, 0, 1, 1, 0]
    assert result["rewards"] == [1, 0, 0, 1, 0, 1, 1, 0]
    assert rewards[-1] == int(correct_external)
