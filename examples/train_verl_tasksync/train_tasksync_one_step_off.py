# Copyright 2025 GEM Team
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

"""Hydra entry for one-step-off TaskSync training.

Mirrors ``examples/train_verl_tasksync/train_tasksync.py`` but:
  - swaps in ``TaskSyncOneStepOffTrainer``
  - uses Role.Actor (with DetachActorWorker) instead of Role.ActorRollout
  - reads rollout pool spec from ``config.rollout.{nnodes,n_gpus_per_node}``
"""

import asyncio
import logging
import os
import socket
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.experimental.separation.utils import (
    create_resource_pool_manager,
    create_role_worker_mapping,
)
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device, is_cuda_available

from .tasksync_trainer_one_step_off import TaskSyncOneStepOffTrainer

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class TaskSyncOneStepOffRunner:
    """Ray remote driver. Wraps trainer.fit() in asyncio.run because
    OneStepOffRayTrainer.fit is async."""

    def run(self, config):
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Mirror one-step-off main_ppo: surface rollout pool spec to the
        # AgentLoopManager via actor_rollout_ref.rollout.{nnodes,n_gpus_per_node}.
        config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
        config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node

        # Force hybrid_engine=False for resource separation. We do this here
        # as a guard so a stray override in the shell can't silently break
        # the trainer's assumptions.
        config.actor_rollout_ref.hybrid_engine = False

        # Worker mapping: actor uses DetachActorWorker (training-only); rollout
        # is managed by AgentLoopManager itself (no Ray worker group needed).
        # We don't use create_role_worker_mapping from separation.utils because
        # it pulls in TrainingWorker for critic; TaskSync has no critic.
        role_worker_mapping = {Role.Actor: ray.remote(DetachActorWorker)}
        if need_reference_policy(config):
            role_worker_mapping[Role.RefPolicy] = ray.remote(DetachActorWorker)

        # Resource pools: actor on `trainer.*`, rollout pool created internally
        # by AgentLoopManager via `actor_rollout_ref.rollout.{nnodes,n_gpus_per_node}`.
        resource_pool_spec = {
            "trainer_pool": (
                [config.trainer.n_gpus_per_node] * config.trainer.nnodes
            ),
        }
        mapping = {Role.Actor: "trainer_pool"}
        if need_reference_policy(config):
            mapping[Role.RefPolicy] = "trainer_pool"

        if config.reward.reward_model.enable and config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("reward.reward_model.n_gpus_per_node must be > 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("reward.reward_model.nnodes must be > 0")
            resource_pool_spec["reward_pool"] = (
                [config.reward.reward_model.n_gpus_per_node]
                * config.reward.reward_model.nnodes
            )
            mapping[Role.RewardModel] = "reward_pool"
        elif config.reward.reward_model.enable:
            mapping[Role.RewardModel] = "trainer_pool"
        else:
            # Schema sometimes expects these set; mirror trainer pool sizing.
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        trainer = TaskSyncOneStepOffTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            device_name=config.trainer.get("device", "cuda"),
        )
        trainer.init_workers()
        asyncio.run(trainer.fit())


def run_training(config) -> None:
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_kwargs", {}).get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create(
            {**ray_init_kwargs, "runtime_env": runtime_env}
        )
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if (
        is_cuda_available
        and OmegaConf.select(config, "global_profiler.tool") == "nsys"
        and OmegaConf.select(config, "global_profiler.steps") is not None
        and len(OmegaConf.select(config, "global_profiler.steps", default=[])) > 0
    ):
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskSyncOneStepOffRunner.options(
            runtime_env={"nsight": nsight_options}
        ).remote()
    else:
        runner = TaskSyncOneStepOffRunner.remote()

    ray.get(runner.run.remote(config))

    timeline_json_file = config.get("ray_kwargs", {}).get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@hydra.main(
    config_path="./config",
    config_name="tasksync_grpo_megatron_one_step_off",
    version_base=None,
)
def main(config):
    run_training(config)


if __name__ == "__main__":
    main()
