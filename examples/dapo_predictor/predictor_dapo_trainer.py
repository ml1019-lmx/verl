"""Recipe-side DAPO trainer with predictor-driven rollout reordering."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, apply_kl_penalty, compute_advantage
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum, TokenizationSanityCheckModeEnum

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

    @staticmethod
    def _pad_maybe_mrope(tensors: list[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
        if not tensors:
            raise ValueError("Cannot pad an empty tensor list.")

        if tensors[0].dim() == 1:
            return pad_sequence(tensors, batch_first=True, padding_value=pad_value)

        if tensors[0].dim() == 2:
            channels = tensors[0].shape[0]
            max_len = max(t.shape[-1] for t in tensors)
            padded = torch.full((len(tensors), channels, max_len), pad_value, dtype=tensors[0].dtype)
            for i, tensor in enumerate(tensors):
                padded[i, :, : tensor.shape[-1]] = tensor
            return padded

        raise ValueError(f"Unsupported tensor rank for padding: {tensors[0].dim()}")

    def _hydrate_gen_batch_model_inputs(self, gen_batch: DataProto) -> DataProto:
        if {"input_ids", "attention_mask", "position_ids"}.issubset(set(gen_batch.batch.keys())):
            return gen_batch

        processing_class = self.processor if self.processor is not None else self.tokenizer
        rollout_cfg = self.config.actor_rollout_ref.rollout
        data_cfg = self.config.data
        max_prompt_len = int(data_cfg.get("max_prompt_length", 8192))
        max_response_len = int(rollout_cfg.get("response_length", 8192))
        use_inference_chat_template = bool(rollout_cfg.get("use_inference_chat_template", True))
        tokenization_mode = TokenizationSanityCheckModeEnum(
            rollout_cfg.get("tokenization_sanity_check_mode", TokenizationSanityCheckModeEnum.STRICT.value)
        )

        messages_list = gen_batch.non_tensor_batch.get("messages")
        if messages_list is None:
            return gen_batch

        tool_schemas_list = gen_batch.non_tensor_batch.get("tool_schemas")
        multi_modal_data_list = gen_batch.non_tensor_batch.get("multi_modal_data")

        input_ids_list: list[torch.Tensor] = []
        attention_mask_list: list[torch.Tensor] = []
        position_ids_list: list[torch.Tensor] = []
        for i, messages in enumerate(messages_list):
            request = AsyncRolloutRequest.model_validate(
                {
                    "batch_data_id": i,
                    "rollout_offset": 0,
                    "request_id": f"predictor-{i}",
                    "state": AsyncRolloutRequestStateEnum.PENDING,
                    "messages": messages,
                    "tool_schemas": tool_schemas_list[i] if tool_schemas_list is not None else None,
                    "multi_modal_data": multi_modal_data_list[i] if multi_modal_data_list is not None else None,
                    "reward_scores": {},
                    "max_prompt_len": max_prompt_len,
                    "max_response_len": max_response_len,
                    "use_inference_chat_template": use_inference_chat_template,
                    "tokenization_sanity_check_mode": tokenization_mode,
                    "processing_class": processing_class,
                }
            )
            input_ids_list.append(request.input_ids.squeeze(0))
            attention_mask_list.append(request.attention_mask.squeeze(0))
            position_ids_list.append(request.position_ids.squeeze(0))

        gen_batch.batch["input_ids"] = self._pad_maybe_mrope(input_ids_list, pad_value=self.tokenizer.pad_token_id)
        gen_batch.batch["attention_mask"] = self._pad_maybe_mrope(attention_mask_list, pad_value=0)
        gen_batch.batch["position_ids"] = self._pad_maybe_mrope(position_ids_list, pad_value=0)
        return gen_batch

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
                gen_batch = self._hydrate_gen_batch_model_inputs(gen_batch)
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
