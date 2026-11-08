import numpy as np
import unittest

from craftax.sea.goal_ranking_analysis import decision_metrics, sequence_metrics


class TestGoalRankingAnalysis(unittest.TestCase):
    def test_tie_break_is_not_a_strict_ranking_flip(self):
        result = decision_metrics([[1, 1, 0]], [[0, 0.2, 0]], [[True] * 3], 0.8)
        assert result["argmax_changed"][0]
        assert result["tie_only_argmax_change"][0]
        assert not result["strict_top_flip"][0]
        assert result["static_top_tie_count"][0] == 2


    def test_lower_static_candidate_can_overtake_static_top_set(self):
        result = decision_metrics([[1, 1, 0]], [[0, 0, 2]], [[True] * 3], 0.8)
        assert result["augmented_argmax"][0] == 2
        assert result["strict_top_flip"][0]
        assert result["reversed_pairs"][0] == 2
        assert result["ordered_pairs"][0] == 2


    def test_masked_extreme_does_not_change_ranking_or_probabilities(self):
        result = decision_metrics([[1, 0, 100]], [[0, 0, 1e6]], [[True, True, False]], 0.8)
        assert result["augmented_argmax"][0] == 0
        assert not result["argmax_changed"][0]
        np.testing.assert_allclose(result["augmented_probabilities"], [[0.73105858, 0.26894142, 0]])


    def test_beta_zero_control_is_identical_but_not_always_uniform(self):
        result = decision_metrics([[1, 0], [0, 0]], [[-100, 100], [100, -100]], [[True] * 2] * 2, 0)
        assert not result["argmax_changed"].any()
        np.testing.assert_array_equal(result["probability_tv"], 0)
        assert result["static_uniform_tv"][0] > 0
        assert result["static_uniform_tv"][1] == 0


    def test_empty_candidate_set_is_rejected(self):
        with self.assertRaises(ValueError):
            decision_metrics([[0, 0]], [[0, 0]], [[False, False]], 0.8)


    def test_sequence_comparison_preserves_repetitions_and_length_changes(self):
        assert sequence_metrics([1, 1, 2], [1, 1, 2])["identical"]
        result = sequence_metrics([1, 1, 2], [1, 2])
        assert result["common_prefix_length"] == 1
        assert result["edit_distance"] == 1
        assert result["first_different_selection"] == 1
        assert sequence_metrics([], [1])["normalized_edit_distance"] == 1
