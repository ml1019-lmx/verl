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
"""vLLM-Ascend PD-disaggregated replica: single-node 1 prefill + N decode servers."""

import asyncio
import logging
from dataclasses import replace as _dc_replace
from typing import Optional

import ray
from ray.actor import ActorHandle

from verl.utils.device import get_resource_name, is_torch_npu_available
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.pd_utils import resolve_vllm_pd_decode_tp, vllm_pd_world_size
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class vLLMPDReplica(vLLMReplica):
    """Replica that runs vLLM-Ascend in prefill/decode disaggregated mode."""

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank,
            config,
            model_config,
            gpus_per_node,
            is_reward_model,
            is_teacher_model,
            name_suffix,
        )
        disagg = self.config.disaggregation
        assert disagg.enabled, "vLLMPDReplica requires rollout.disaggregation.enabled=True"

        if disagg.prefill_replicas != 1:
            raise NotImplementedError(f"prefill_replicas=1 only (got {disagg.prefill_replicas})")
        if self.config.data_parallel_size != 1:
            raise NotImplementedError(f"data_parallel_size=1 only (got {self.config.data_parallel_size})")
        if self.config.pipeline_model_parallel_size != 1:
            raise NotImplementedError(
                f"pipeline_model_parallel_size=1 only (got {self.config.pipeline_model_parallel_size})"
            )
        if disagg.transfer_backend != "mooncake":
            raise ValueError(
                f"vLLM PD requires disaggregation.transfer_backend='mooncake', got {disagg.transfer_backend!r}"
            )

        self._n_decode = disagg.decode_replicas
        self._prefill_tp = self.config.tensor_model_parallel_size
        self._decode_tp = resolve_vllm_pd_decode_tp(self.config)
        pd_world_size = vllm_pd_world_size(self.config)
        if pd_world_size > gpus_per_node:
            raise NotImplementedError(
                f"vLLM PD replica needs {pd_world_size} devices but gpus_per_node={gpus_per_node}; "
                "cross-node PD is not supported yet."
            )

        self.world_size = pd_world_size
        self.gpus_per_replica_node = min(self.gpus_per_node, self.world_size)
        assert self.world_size % self.gpus_per_replica_node == 0
        self.nnodes = self.world_size // self.gpus_per_replica_node

        self._prefill_servers: list[ActorHandle] = []
        self._decode_servers: list[ActorHandle] = []
        self._prefill_server_address: Optional[str] = None
        self._decode_server_addresses: list[str] = []

    async def launch_servers(self):
        """Launch one prefill server and N decode servers on a single node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )
        if not is_torch_npu_available(check_device=False):
            raise NotImplementedError("vLLM PD disaggregation is NPU/vLLM-Ascend only.")
        if self.nnodes != 1:
            raise NotImplementedError("vLLM PD disaggregation currently supports a single node only.")

        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                    )
                )
                for worker in self.workers
            ]
        )
        worker_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]

        prefill_end = self._prefill_tp
        prefill_server = await self._launch_one(
            role="prefill",
            pd_server_index=0,
            workers=self.workers[:prefill_end],
            node_id=worker_node_ids[0],
            visible_devices=",".join(worker_visible_devices[:prefill_end]),
            tp=self._prefill_tp,
            actor_name=f"vllm_server_{self.replica_rank}_0{self.name_suffix}",
        )
        self._prefill_servers = [prefill_server]

        prefill_address, prefill_port = await prefill_server.get_server_address.remote()
        self._prefill_server_address = self._format_address(prefill_address, prefill_port)

        self._decode_servers = []
        self._decode_server_addresses = []
        for decode_index in range(self._n_decode):
            start = self._prefill_tp + decode_index * self._decode_tp
            end = start + self._decode_tp
            decode_server = await self._launch_one(
                role="decode",
                pd_server_index=decode_index,
                workers=self.workers[start:end],
                node_id=worker_node_ids[start],
                visible_devices=",".join(worker_visible_devices[start:end]),
                tp=self._decode_tp,
                actor_name=f"vllm_server_decode_{self.replica_rank}_{decode_index}{self.name_suffix}",
            )
            self._decode_servers.append(decode_server)

            decode_address, decode_port = await decode_server.get_server_address.remote()
            self._decode_server_addresses.append(self._format_address(decode_address, decode_port))

        self.servers = list(self._prefill_servers) + list(self._decode_servers)
        self._server_handle = prefill_server
        self._server_address = self._prefill_server_address

        await prefill_server.set_pd_peer.remote(list(self._decode_servers))
        logger.info(
            "vLLMPDReplica rank=%s launched: prefill=%s, decodes=[%s]",
            self.replica_rank,
            self._prefill_server_address,
            ", ".join(self._decode_server_addresses),
        )

    async def _launch_one(
        self,
        role: str,
        pd_server_index: int,
        workers: list[ActorHandle],
        node_id: str,
        visible_devices: str,
        tp: int,
        actor_name: str,
    ) -> ActorHandle:
        pool_config = _dc_replace(self.config, tensor_model_parallel_size=tp)
        server = self.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                    "NCCL_CUMEM_ENABLE": "0",
                }
            },
            name=actor_name,
            max_concurrency=self.max_concurrency,
        ).remote(
            config=pool_config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            workers=workers,
            replica_rank=self.replica_rank,
            node_rank=0,
            gpus_per_node=tp,
            nnodes=1,
            cuda_visible_devices=visible_devices,
            disaggregation_role=role,
            pd_server_index=pd_server_index,
            pd_prefill_tp=self._prefill_tp,
            pd_decode_tp=self._decode_tp,
        )
        await server.launch_server.remote(master_address=None, master_port=None, dp_rpc_port=None)
        return server

    @staticmethod
    def _format_address(address: str, port: int) -> str:
        return f"[{address}]:{port}" if is_valid_ipv6_address(address) else f"{address}:{port}"
