# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helpers for vLLM-Ascend prefill/decode disaggregation."""

from typing import Literal, Optional

from verl.workers.config import RolloutConfig

VLLM_PD_DEFAULT_KV_PORT = 23010
VLLM_PD_ROLE_PREFILL = "prefill"
VLLM_PD_ROLE_DECODE = "decode"


def resolve_vllm_pd_decode_tp(config: RolloutConfig) -> int:
    """Resolve decode TP, defaulting to the prefill/server TP."""
    disagg = config.disaggregation
    if disagg.decode_tensor_model_parallel_size is not None:
        return disagg.decode_tensor_model_parallel_size
    return config.tensor_model_parallel_size


def vllm_pd_world_size(config: RolloutConfig) -> int:
    """Return total workers for one vLLM PD replica: 1P + N decode servers."""
    prefill_tp = config.tensor_model_parallel_size
    decode_tp = resolve_vllm_pd_decode_tp(config)
    return prefill_tp + config.disaggregation.decode_replicas * decode_tp


def resolve_vllm_pd_kv_port(config: RolloutConfig, pd_server_index: int) -> int:
    """Return the Mooncake KV-transfer base port for a PD server."""
    base_port = config.disaggregation.kv_port or VLLM_PD_DEFAULT_KV_PORT
    return base_port + pd_server_index * config.disaggregation.kv_port_stride


def build_vllm_pd_kv_transfer_config(
    config: RolloutConfig,
    role: Literal["prefill", "decode"],
    pd_server_index: int = 0,
    engine_id: Optional[str] = None,
) -> dict:
    """Build vLLM/vLLM-Ascend kv_transfer_config for a PD server."""
    if role not in (VLLM_PD_ROLE_PREFILL, VLLM_PD_ROLE_DECODE):
        raise ValueError(f"Unsupported vLLM PD role: {role!r}")

    prefill_tp = config.tensor_model_parallel_size
    decode_tp = resolve_vllm_pd_decode_tp(config)
    kv_role = "kv_producer" if role == VLLM_PD_ROLE_PREFILL else "kv_consumer"

    kv_config = {
        "kv_connector": config.disaggregation.kv_connector,
        "kv_role": kv_role,
        "kv_buffer_device": "npu",
        "kv_port": resolve_vllm_pd_kv_port(config, pd_server_index),
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 1, "tp_size": prefill_tp},
            "decode": {"dp_size": 1, "tp_size": decode_tp},
        },
    }
    if engine_id is not None:
        kv_config["engine_id"] = engine_id
    return kv_config


def vllm_pd_role_for_rank(config: RolloutConfig, rollout_rank: int) -> tuple[str, int, int]:
    """Map a rollout rank to (role, server_index, tp_local_rank)."""
    prefill_tp = config.tensor_model_parallel_size
    decode_tp = resolve_vllm_pd_decode_tp(config)
    if rollout_rank < prefill_tp:
        return VLLM_PD_ROLE_PREFILL, 0, rollout_rank

    decode_offset = rollout_rank - prefill_tp
    decode_index = decode_offset // decode_tp
    decode_local_rank = decode_offset % decode_tp
    if decode_index >= config.disaggregation.decode_replicas:
        raise ValueError(
            f"rollout_rank={rollout_rank} outside vLLM PD world size {vllm_pd_world_size(config)}"
        )
    return VLLM_PD_ROLE_DECODE, decode_index, decode_local_rank
