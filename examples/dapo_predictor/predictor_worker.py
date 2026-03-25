"""Recipe-side worker extensions for predictor-driven prompt reordering."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from codetiming import Timer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from verl import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.attention_utils import index_first_axis, rearrange, unpad_input
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.profiler import DistProfiler
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker


class PredictorDataParallelPPOActor(DataParallelPPOActor):
    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config=config, actor_module=actor_module, actor_optimizer=actor_optimizer)
        hidden_size = getattr(getattr(actor_module, "config", None), "hidden_size", None)
        if hidden_size is None and hasattr(actor_module, "module"):
            hidden_size = getattr(getattr(actor_module.module, "config", None), "hidden_size", None)
        hidden_size = hidden_size or 4096
        self.predictor_scorer = nn.Linear(hidden_size, 1, bias=False).to(next(actor_module.parameters()).device)
        if torch.distributed.is_initialized():
            for param in self.predictor_scorer.parameters():
                torch.distributed.broadcast(param.data, src=0)

    def extract_hidden_states(self, data: DataProto) -> torch.Tensor:
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch
        select_keys = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            batches = data.split(micro_batch_size)
            batch_idx_list = None
        hidden_states = []
        for micro_batch in batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch}
            if "multi_modal_inputs" in micro_batch.non_tensor_batch:
                model_inputs["multi_modal_inputs"] = micro_batch.non_tensor_batch["multi_modal_inputs"]
            with torch.no_grad():
                hidden_states.append(self._forward_predictor_micro_batch(model_inputs))
        all_hidden_states = torch.cat(hidden_states, dim=0)
        if batch_idx_list is not None:
            all_hidden_states = restore_dynamic_batch(all_hidden_states, batch_idx_list)
        return all_hidden_states

    def _forward_predictor_micro_batch(self, micro_batch: dict[str, Any]) -> torch.Tensor:
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # mrope path
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(getattr(self.actor_module, "module", self.actor_module).config, "vision_config")
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                else:
                    pad_size = 0

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                    output_hidden_states=True,
                    **multi_modal_inputs,
                )
                hidden_rmpad = output.hidden_states[-1].squeeze(0)

                if self.use_ulysses_sp:
                    hidden_rmpad = gather_outputs_and_unpad(
                        hidden_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )

                seq_hidden = torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    hidden_rmpad.shape[-1],
                    device=hidden_rmpad.device,
                    dtype=hidden_rmpad.dtype,
                )
                seq_hidden = seq_hidden.view(-1, hidden_rmpad.shape[-1])
                seq_hidden[indices] = hidden_rmpad
                seq_hidden = seq_hidden.view(input_ids.shape[0], input_ids.shape[1], hidden_rmpad.shape[-1])
                last_token_idx = attention_mask.sum(dim=-1) - 1
                return seq_hidden[torch.arange(seq_hidden.shape[0], device=seq_hidden.device), last_token_idx]

            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
                **multi_modal_inputs,
            )
            hidden = output.hidden_states[-1]
            last_token_idx = attention_mask.sum(dim=-1) - 1
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), last_token_idx]

    @staticmethod
    def listmle_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        random_indices = torch.randperm(y_pred.shape[-1], device=y_pred.device)
        y_pred_shuffled = y_pred[:, random_indices].float()
        y_true_shuffled = y_true[:, random_indices].float()
        _, indices = y_true_shuffled.sort(descending=True, dim=-1)
        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
        observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
        return torch.mean(torch.sum(observation_loss, dim=1))


class PredictorAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def init_model(self):
        super().init_model()
        if self._is_actor:
            self.actor = PredictorDataParallelPPOActor(
                config=self.actor.config,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

    def _predictor_cfg(self):
        return self.config.trainer.get("predictor_reorder", {})

    def _actor_params_are_offloaded(self) -> bool:
        return next(self.actor_module_fsdp.parameters()).device.type == "cpu"

    def _sync_predictor_scorer_device(self):
        actor_device = next(self.actor_module_fsdp.parameters()).device
        if next(self.actor.predictor_scorer.parameters()).device != actor_device:
            self.actor.predictor_scorer = self.actor.predictor_scorer.to(actor_device)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        keep_actor_loaded = bool(data.meta_info.get("predictor_keep_actor_loaded", False))
        if not (keep_actor_loaded and self._is_offload_param):
            return super().update_actor(data)

        load_fsdp_model_to_gpu(self.actor_module_fsdp)
        self._sync_predictor_scorer_device()

        original_is_offload_param = self._is_offload_param
        self._is_offload_param = False
        try:
            return super().update_actor(data)
        finally:
            self._is_offload_param = original_is_offload_param

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="purple", role="predictor_compute_score")
    def compute_predictor_score(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        self._sync_predictor_scorer_device()

        data = data.to(get_device_id())
        data.meta_info["micro_batch_size"] = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz

        n = self.config.rollout.n
        batch_size = len(data)
        sample_indices = list(range(0, batch_size, n))
        sampled_non_tensors = {}
        for key, val in data.non_tensor_batch.items():
            sampled_non_tensors[key] = val[sample_indices] if isinstance(val, np.ndarray) else val
        sampled = DataProto(
            batch=data.batch[sample_indices],
            non_tensor_batch=sampled_non_tensors,
            meta_info=data.meta_info.copy(),
        )

        with self.ulysses_sharding_manager:
            hidden_states = self.actor.extract_hidden_states(sampled)
        scores = self.actor.predictor_scorer(hidden_states).squeeze(-1)

        predictor_scores = torch.zeros(batch_size, device=scores.device, dtype=scores.dtype)
        for i, sample_idx in enumerate(sample_indices):
            predictor_scores[sample_idx : min(sample_idx + n, batch_size)] = scores[i]

        output = DataProto.from_dict(tensors={"predictor_scores": predictor_scores}).to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="orange", role="predictor_update")
    def update_predictor(self, prompt_batch: DataProto, response_batch: DataProto):
        assert self._is_actor
        cfg = self._predictor_cfg()
        if not cfg.get("enable", False):
            return DataProto(meta_info={"metrics": {}})

        loaded_actor_for_predictor = self._is_offload_param and self._actor_params_are_offloaded()
        if loaded_actor_for_predictor:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        self._sync_predictor_scorer_device()

        prompt_batch = prompt_batch.to(get_device_id())
        prompt_batch.meta_info["micro_batch_size"] = self.config.ref.log_prob_micro_batch_size_per_gpu
        prompt_batch.meta_info["temperature"] = self.config.rollout.temperature
        prompt_batch.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        prompt_batch.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz

        n = self.config.rollout.n
        sample_indices = list(range(0, len(prompt_batch), n))
        sampled_non_tensors = {}
        for key, val in prompt_batch.non_tensor_batch.items():
            sampled_non_tensors[key] = val[sample_indices] if isinstance(val, np.ndarray) else val
        sampled_prompt = DataProto(
            batch=prompt_batch.batch[sample_indices],
            non_tensor_batch=sampled_non_tensors,
            meta_info=prompt_batch.meta_info.copy(),
        )

        with self.ulysses_sharding_manager:
            hidden_states = self.actor.extract_hidden_states(sampled_prompt)

        response_batch = response_batch.to(get_device_id())
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        response_lengths = (response_batch.batch["responses"] != pad_token_id).sum(dim=1)
        response_lengths = response_lengths.view(-1, n).max(dim=1).values.float()

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            gathered_hidden = [torch.empty_like(hidden_states) for _ in range(world_size)]
            gathered_lengths = [torch.empty_like(response_lengths) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_hidden, hidden_states)
            torch.distributed.all_gather(gathered_lengths, response_lengths)
            hidden_states = torch.cat(gathered_hidden, dim=0)
            response_lengths = torch.cat(gathered_lengths, dim=0)

        predictor = self.actor.predictor_scorer.float()
        optimizer = torch.optim.AdamW(
            predictor.parameters(),
            lr=cfg.get("lr", 3e-5),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        dataset = TensorDataset(hidden_states.float(), response_lengths.float())
        dataloader = DataLoader(dataset, batch_size=cfg.get("batch_size", 32), shuffle=True, drop_last=False)
        epochs = cfg.get("epochs", 10)

        metrics = {}
        with Timer(name="predictor_update", logger=None) as timer:
            for _ in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                for batch_hidden, batch_lengths in dataloader:
                    preds = predictor(batch_hidden).squeeze(-1).unsqueeze(0)
                    labels = batch_lengths.unsqueeze(0)
                    loss = self.actor.listmle_loss(preds, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1
                metrics["predictor/final_loss"] = epoch_loss / max(num_batches, 1)
        metrics["predictor/epochs"] = epochs
        metrics["predictor/update_time_s"] = timer.last
        metrics["predictor/total_samples"] = len(dataset)

        output = DataProto(meta_info={"metrics": metrics}).to("cpu")
        if loaded_actor_for_predictor:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        return output
