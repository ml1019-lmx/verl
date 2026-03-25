from examples.dapo_predictor.predictor_utils import snake_sort_indices


def test_snake_sort_indices_groups_scores_by_prompt_and_snakes_across_dp():
    scores = [0.9, 0.9, 0.2, 0.2, 0.7, 0.7, 0.1, 0.1]
    indices = snake_sort_indices(scores, n_samples_per_prompt=2, dp_world_size=2)
    assert indices == [0, 1, 6, 7, 4, 5, 2, 3]
