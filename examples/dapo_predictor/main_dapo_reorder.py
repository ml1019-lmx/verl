"""Backward-compatible alias for predictor-driven DAPO reorder entrypoint."""

import hydra
import ray

from recipe.dapo_predictor.main_dapo_predictor_reorder import PredictorDAPOTaskRunner
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device


@hydra.main(config_path="../dapo/config", config_name="dapo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(PredictorDAPOTaskRunner))


if __name__ == "__main__":
    main()
