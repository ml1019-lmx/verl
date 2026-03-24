"""Recipe-side DAPO trainer with predictor-driven rollout reordering."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, apply_kl_penalty, compute_advantage
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer

from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

from .predictor_utils import snake_sort_indices


class PredictorRayDAPOTrainer(RayDAPOTrainer):
    """DAPO trainer that only injects the predictor-specific reorder steps.

    Most heavy lifting is still reused from the current `RayDAPOTrainer` / `RayPPOTrainer` stack:
    reward computation, KL/ref/value computation, actor/critic updates, checkpointing, metrics,
    and rollout manager orchestration all stay on the upstream path.
    """

    def _predictor_cfg(self):
        return self.config.trainer.get("predictor_reorder", {})

    def _predictor_enabled(self) -> bool:
        return self._predictor_cfg().get("enable", False)

    def _build_predictor_order(self, gen_batch: DataProto) -> torch.Tensor:
        predictor_scores = self.actor_rollout_wg.compute_predictor_score(gen_batch)
        gen_batch = gen_batch.union(predictor_scores)
        dp_world_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        return torch.tensor(
            snake_sort_indices(
                gen_batch.batch["predictor_scores"].tolist(),
                n_samples_per_prompt=self.config.actor_rollout_ref.rollout.n,
                dp_world_size=dp_world_size,
            ),
            dtype=torch.long,
        )

    def _apply_predictor_order(self, batch: DataProto, predictor_order: torch.Tensor | None) -> DataProto:
        if predictor_order is not None:
            batch.reorder(predictor_order)
        return batch

    def _repeat_and_tag_uid(self, batch: DataProto) -> DataProto:
        batch.non_tensor_batch["uid"] = np.array([str(i) for i in range(len(batch.batch))], dtype=object)
        return batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

    def _maybe_update_predictor(self, gen_batch: DataProto, batch: DataProto, metrics, timing_raw):
        with marked_timer("update_predictor", timing_raw, "orange"):
            predictor_output = self.actor_rollout_wg.update_predictor(gen_batch, batch)
        metrics.update(reduce_metrics(predictor_output.meta_info.get("metrics", {})))

    def fit(self):
        if not self._predictor_enabled():
            return super().fit()

        from collections import defaultdict
        from pprint import pprint

        from omegaconf import OmegaConf
        from tqdm import tqdm

        from verl.trainer.ppo.reward import extract_reward
        from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
        from verl.utils.rollout_skip import RolloutSkip
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        self.max_steps_duration = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps if self.config.global_profiler.steps is not None else False
        )
        timing_raw = defaultdict(float)
        current_epoch = self.global_steps // len(self.train_dataloader)

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(curr_step_profile)

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                gen_batch = self._get_gen_batch(new_batch)
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("predictor_score", timing_raw, "purple"):
                        predictor_order = self._build_predictor_order(gen_batch)
                        self._apply_predictor_order(gen_batch, predictor_order)

                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            new_batch = new_batch.union(gen_baseline_output)
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self._compute_reward_colocate(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = extract_reward(new_batch)
                            new_batch.batch["reward_baselines"] = reward_baseline_tensor.sum(dim=-1)
                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                    batch = self._repeat_and_tag_uid(new_batch)
                    self._apply_predictor_order(batch, predictor_order)
                    batch = batch.union(gen_batch_output)

                    if self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    with marked_timer("reward", timing_raw, "yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)
                        batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)
                    with marked_timer("adv", timing_raw, "brown"):
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                            config=self.config.algorithm,
                        )
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self._update_critic(batch)
                        metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        batch.meta_info["predictor_keep_actor_loaded"] = True
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self._update_actor(batch)
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                        ):
                            with marked_timer("save_checkpoint", timing_raw, "grey"):
                                self._save_checkpoint()

                    self._maybe_update_predictor(gen_batch, batch, metrics, timing_raw)

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=self.n_gpus))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                if self.global_steps >= self.total_training_steps:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    return
                self.global_steps += 1
                self.gen_steps += 1
