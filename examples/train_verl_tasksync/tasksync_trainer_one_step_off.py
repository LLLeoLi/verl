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

"""One-step-off-policy variant of TaskSyncTrainer.

Inherits from ``OneStepOffRayTrainer`` (separate actor and rollout resource
pools, async overlap) but ports the TaskSync-specific data path:

    - prompts come from ``TaskSyncDataset.get_batch(batch_idx)``, not from a
      Ray data loader
    - rewards live in ``output.non_tensor_batch["episode_reward"]`` directly,
      so we set ``token_level_scores`` ourselves
    - validation, save/load, and experience dumping reuse the original
      TaskSync implementations
"""

import json
import logging
import os
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.utils.data
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict
from torch.utils.data import SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import (
    ResourcePoolManager,
    compute_response_mask,
)
from verl.utils.debug import marked_timer
from verl.utils.ray_utils import auto_await

from .tasksync_dataset import TaskSyncDataset

logger = logging.getLogger(__name__)


class _DummyPromptDataset(torch.utils.data.Dataset):
    """Placeholder dataset; real prompts come from TaskSyncDataset.get_batch."""

    def __init__(self, size: int = 1):
        self.size = max(1, size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        # Carry the dataset index through the loader so the async generator
        # knows which TaskSyncDataset batch to pull.
        return {"_tasksync_batch_idx": idx}


def _identity_collate(batch):
    # batch is a list of dicts with one key; surface the first index.
    return batch[0] if batch else {"_tasksync_batch_idx": 0}


class TaskSyncOneStepOffTrainer(OneStepOffRayTrainer):
    """TaskSync trainer with one-step-off-policy async rollout."""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict,
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls=RayWorkerGroup,
        processor=None,
        device_name=None,
    ):
        env_cfg = config.actor_rollout_ref.env
        tasks_dir = env_cfg.tasks_dir
        val_ratio = float(env_cfg.get("val_ratio", 0.0))
        seed = config.get("seed", 42)

        self.ptc_mode = env_cfg.get("ptc_mode", "ptc")
        assert self.ptc_mode in ("ptc", "no-ptc", "mixed"), (
            f"ptc_mode must be 'ptc', 'no-ptc', or 'mixed', got '{self.ptc_mode}'"
        )
        self.ptc_desc = env_cfg.get("ptc_desc", "rich")
        if self.ptc_mode == "mixed":
            old_mini = config.actor_rollout_ref.actor.ppo_mini_batch_size
            with open_dict(config):
                config.actor_rollout_ref.actor.ppo_mini_batch_size = old_mini * 2
            logger.info(
                f"[ptc_mode=mixed] doubled ppo_mini_batch_size: {old_mini} -> {old_mini * 2}"
            )

        common_kwargs = dict(
            tasks_dir=tasks_dir,
            batch_size=env_cfg.get("batch_size", 8),
            group_size=env_cfg.get("group_size", 32),
            max_tool_calls=env_cfg.get("max_tool_calls", 50),
            filter_pattern=env_cfg.get("filter_pattern", None),
            seed=seed,
            validate_on_discovery=env_cfg.get("validate_on_discovery", False),
            val_ratio=val_ratio,
        )

        if val_ratio > 0:
            self.planning_dataset = TaskSyncDataset(split="train", **common_kwargs)
            try:
                self.val_planning_dataset: Optional[TaskSyncDataset] = TaskSyncDataset(
                    split="val", **common_kwargs
                )
            except ValueError:
                self.val_planning_dataset = None
        else:
            self.planning_dataset = TaskSyncDataset(split=None, **common_kwargs)
            self.val_planning_dataset = None

        steps_per_epoch = max(1, len(self.planning_dataset))
        val_steps_per_epoch = (
            max(1, len(self.val_planning_dataset)) if self.val_planning_dataset is not None else 1
        )
        self._tasksync_train_size = steps_per_epoch
        self._tasksync_val_size = val_steps_per_epoch

        # OneStepOffRayTrainer's __init__ expects: config, tokenizer, role_worker_mapping,
        # resource_pool_manager, ray_worker_group_cls, processor, train_dataset, val_dataset,
        # collate_fn, train_sampler, device_name
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            train_dataset=_DummyPromptDataset(steps_per_epoch),
            val_dataset=_DummyPromptDataset(val_steps_per_epoch),
            collate_fn=_identity_collate,
            train_sampler=SequentialSampler(_DummyPromptDataset(steps_per_epoch)),
            device_name=device_name,
        )

        self.dump_experience_every = int(
            config.actor_rollout_ref.get("dump_experience_every", -1)
        )
        self.step_count = 0
        self.actor_id = 0
        self.game_state_save_path = os.path.join(
            self.config.trainer.default_local_dir, "game_states"
        )
        os.makedirs(self.game_state_save_path, exist_ok=True)

        # Used by _run_rollout / _async_gen_next_batch to set reward type per epoch.
        self._effective_reward_type: Optional[str] = None

    # ------------------------------------------------------------------
    # Dataloader: dummy. Real prompt batches come from TaskSyncDataset.
    # ------------------------------------------------------------------
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=False,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                        total_training_steps
                    )
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            logger.warning(f"Could not set total_training_steps in optim config: {e}")

    # ------------------------------------------------------------------
    # Continuous iterator: yields (epoch, batch_idx) instead of (epoch, batch_dict)
    # so _async_gen_next_batch can index TaskSyncDataset directly.
    # ------------------------------------------------------------------
    def _create_continuous_iterator(self):
        steps_per_epoch = max(1, len(self.train_dataloader))
        start_epoch = self.global_steps // steps_per_epoch
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            for batch_idx, _ in enumerate(self.train_dataloader):
                yield epoch, batch_idx

    # ------------------------------------------------------------------
    # Async rollout: build prompts from TaskSyncDataset, await generate.
    # Returns the same 5-tuple shape as OneStepOffRayTrainer expects.
    # ------------------------------------------------------------------
    async def _async_gen_next_batch(self, continuous_iterator):
        try:
            epoch, batch_idx = next(continuous_iterator)
        except StopIteration:
            return None
        except Exception as e:
            print(f"Error in _async_gen_next_batch: {e}")
            return None

        metrics: dict = {}
        timing_raw: dict = {}

        # Apply dense -> binary reward curriculum based on epoch.
        dense_epoch = self.config.trainer.get("dense_epoch", 0)
        if dense_epoch > 0:
            self._effective_reward_type = "dense" if epoch < dense_epoch else "binary"
        else:
            self._effective_reward_type = None

        with marked_timer("generate_async", timing_raw, color="purple"):
            batch = await self._build_and_generate_async(
                dataset=self.planning_dataset,
                batch_index=batch_idx,
                validate=False,
            )

        if batch is None or batch.batch is None or "input_ids" not in batch.batch:
            # Empty or invalid batch — return a None batch_data_future so
            # caller can skip the step gracefully.
            logger.warning(f"Step {self.global_steps}: empty batch from rollout")
            return metrics, timing_raw, epoch, None, None

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        batch.meta_info["global_token_num"] = torch.sum(
            batch.batch["attention_mask"], dim=-1
        ).tolist()

        # `epoch` and `batch_idx` are needed for save_checkpoint state recovery.
        self._current_epoch = epoch
        self._current_batch_idx = batch_idx

        # No reward future — TaskSync's reward is already in non_tensor_batch.
        return metrics, timing_raw, epoch, batch, None

    @auto_await
    async def _build_and_generate_async(
        self, dataset: TaskSyncDataset, batch_index: int, validate: bool = False
    ) -> Optional[DataProto]:
        """TaskSync rollout: build prompts, await AgentLoopManager, post-process.

        Wrapped in @auto_await so callers in sync context (e.g. _validate) can
        invoke it directly while async callers (_async_gen_next_batch) can await."""
        generate_st = time.time()

        env_groups = dataset.get_batch(batch_index)

        if self.ptc_mode == "mixed":
            ptc_assignments = [(True, "ptc"), (False, "noptc")]
        elif self.ptc_mode == "ptc":
            ptc_assignments = [(True, "ptc")]
        else:
            ptc_assignments = [(False, "noptc")]

        raw_prompts = []
        task_uids = []
        for group_idx, group_builder in enumerate(env_groups):
            task_info = group_builder.task_info
            max_tool_calls = group_builder.max_tool_calls
            task_name = task_info.get("task_name", "unknown")
            for enable_ptc, ptc_tag in ptc_assignments:
                task_uid = f"{task_name}_{group_idx}_{ptc_tag}"
                for _ in range(group_builder.num_envs):
                    raw_prompts.append({
                        "task_info": task_info,
                        "max_tool_calls": max_tool_calls,
                        "enable_ptc": enable_ptc,
                        "ptc_desc": self.ptc_desc,
                    })
                    task_uids.append(task_uid)

        if not raw_prompts:
            logger.warning("No tasks to process, returning empty batch")
            return self._create_empty_batch()

        batch_size = len(raw_prompts)
        size_divisor = max(1, self.config.actor_rollout_ref.rollout.agent.num_workers)
        padded_batch_size = ((batch_size + size_divisor - 1) // size_divisor) * size_divisor
        padding_needed = padded_batch_size - batch_size

        if padding_needed > 0:
            raw_prompts.extend([raw_prompts[0]] * padding_needed)
            task_uids.extend([task_uids[0]] * padding_needed)

        non_tensor_batch = {"raw_prompt": np.array(raw_prompts, dtype=object)}

        if self._effective_reward_type is not None:
            non_tensor_batch["reward_type"] = np.array(
                [self._effective_reward_type] * len(raw_prompts), dtype=object
            )

        data = DataProto(
            batch=TensorDict({}, batch_size=[len(raw_prompts)]),
            non_tensor_batch=non_tensor_batch,
            meta_info={
                "global_steps": self.global_steps,
                "validate": validate,
            },
        )

        output = await self.async_rollout_manager.generate_sequences(data)

        if padding_needed > 0 and output.batch is not None:
            for key in list(output.batch.keys()):
                output.batch[key] = output.batch[key][:batch_size]
            output.batch = TensorDict(
                {k: v for k, v in output.batch.items()},
                batch_size=torch.Size([batch_size]),
            )
            if output.non_tensor_batch:
                for key in list(output.non_tensor_batch.keys()):
                    arr = output.non_tensor_batch[key]
                    if isinstance(arr, np.ndarray):
                        output.non_tensor_batch[key] = arr[:batch_size]
            task_uids = task_uids[:batch_size]

        episode_rewards, episode_successes, dense_rewards, episode_turns = [], [], [], []
        rollout_lengths, truncated_flags, ptc_counts = [], [], []
        ptc_error_counts, ptc_error_penalties = [], []
        python_counts, terminal_counts, env_tool_counts = [], [], []
        episode_rewards_raw: list = []
        if output.non_tensor_batch:
            ntb = output.non_tensor_batch
            if "episode_reward" in ntb:
                episode_rewards = ntb["episode_reward"].tolist()
            if "episode_reward_raw" in ntb:
                episode_rewards_raw = ntb["episode_reward_raw"].tolist()
            if "episode_success" in ntb:
                episode_successes = ntb["episode_success"].tolist()
            if "dense_reward" in ntb:
                dense_rewards = ntb["dense_reward"].tolist()
            if "episode_turns" in ntb:
                episode_turns = ntb["episode_turns"].tolist()
            if "rollout_length" in ntb:
                rollout_lengths = ntb["rollout_length"].tolist()
            if "truncated" in ntb:
                truncated_flags = ntb["truncated"].tolist()
            if "programmatic_tool_call_count" in ntb:
                ptc_counts = ntb["programmatic_tool_call_count"].tolist()
            if "programmatic_tool_call_error_count" in ntb:
                ptc_error_counts = ntb["programmatic_tool_call_error_count"].tolist()
            if "ptc_error_penalty" in ntb:
                ptc_error_penalties = ntb["ptc_error_penalty"].tolist()
            if "execute_python_count" in ntb:
                python_counts = ntb["execute_python_count"].tolist()
            if "terminal_count" in ntb:
                terminal_counts = ntb["terminal_count"].tolist()
            if "env_tool_count" in ntb:
                env_tool_counts = ntb["env_tool_count"].tolist()

        if output.batch is not None and "response_mask" in output.batch:
            response_mask = output.batch["response_mask"]
            response_length = response_mask.shape[-1]
            scalar_rewards = (
                [float(r) for r in episode_rewards]
                if episode_rewards
                else [0.0] * batch_size
            )
            token_level_scores = torch.zeros(
                (batch_size, response_length), dtype=torch.float32
            )
            for i, reward in enumerate(scalar_rewards):
                valid_positions = response_mask[i].nonzero()
                if len(valid_positions) > 0:
                    last_pos = valid_positions[-1].item()
                    token_level_scores[i, last_pos] = reward
                else:
                    token_level_scores[i, -1] = reward
            output.batch["token_level_scores"] = token_level_scores
            output.batch["token_level_rewards"] = token_level_scores.clone()
            output.non_tensor_batch["uid"] = np.array(task_uids, dtype=object)
            # Surface reward as a flat float ndarray for downstream consumers
            # that expect non_tensor_batch["reward"].
            output.non_tensor_batch["reward"] = np.array(
                scalar_rewards, dtype=np.float32
            )

        # Per-uid reward grouping for all-zero / all-one group ratios.
        uid2rewards: dict = defaultdict(list)
        if episode_rewards and len(task_uids) == len(episode_rewards):
            for uid, r in zip(task_uids, episode_rewards, strict=False):
                uid2rewards[uid].append(r)
        total_groups = len(uid2rewards)
        all_zero_groups = sum(1 for vals in uid2rewards.values() if all(v == 0 for v in vals))
        all_one_groups = sum(1 for vals in uid2rewards.values() if all(v == 1 for v in vals))
        all_zero_ratio = all_zero_groups / total_groups if total_groups else 0.0
        all_one_ratio = all_one_groups / total_groups if total_groups else 0.0

        output.meta_info["timing"] = {"actor_time": time.time() - generate_st}
        planning_stats = {
            "group_status/total_groups": total_groups,
            "group_status/all_zero_groups": all_zero_groups,
            "group_status/all_zero_ratio": all_zero_ratio,
            "group_status/all_one_groups": all_one_groups,
            "group_status/all_one_ratio": all_one_ratio,
            "group_status/mean_episode_len": float(np.mean(episode_turns)) if episode_turns else 0.0,
            "group_status/mean_rollout_length": float(np.mean(rollout_lengths)) if rollout_lengths else 0.0,
            "group_status/truncated_rollouts": int(sum(truncated_flags)) if truncated_flags else 0,
            "group_status/mean_ptc_count": float(np.mean(ptc_counts)) if ptc_counts else 0.0,
            "group_status/mean_ptc_error_count": (
                float(np.mean(ptc_error_counts)) if ptc_error_counts else 0.0
            ),
            "group_status/mean_python_count": (
                float(np.mean(python_counts)) if python_counts else 0.0
            ),
            "group_status/mean_terminal_count": (
                float(np.mean(terminal_counts)) if terminal_counts else 0.0
            ),
            "group_status/mean_env_tool_count": (
                float(np.mean(env_tool_counts)) if env_tool_counts else 0.0
            ),
            "reward/score": float(np.mean(dense_rewards)) if dense_rewards else 0.0,
            "reward/score_min": float(np.min(dense_rewards)) if dense_rewards else 0.0,
            "reward/score_max": float(np.max(dense_rewards)) if dense_rewards else 0.0,
            "reward/zero_reward_ratio": (
                dense_rewards.count(0) / len(dense_rewards) if dense_rewards else 1.0
            ),
            "reward/episode_reward_mean": (
                float(np.mean(episode_rewards)) if episode_rewards else 0.0
            ),
            "reward/episode_reward_raw_mean": (
                float(np.mean(episode_rewards_raw)) if episode_rewards_raw else 0.0
            ),
            "reward/mean_ptc_error_penalty": (
                float(np.mean(ptc_error_penalties)) if ptc_error_penalties else 0.0
            ),
            "reward/success_rate": (
                float(np.mean(episode_successes)) if episode_successes else 0.0
            ),
        }
        output.meta_info["planning_stats"] = planning_stats

        if not validate:
            logger.info(
                f"Rollout done: bsz={batch_size}, "
                f"success_rate={planning_stats['reward/success_rate']:.2%}, "
                f"score={planning_stats['reward/score']:.4f}, "
                f"episode_reward={planning_stats['reward/episode_reward_mean']:.4f}, "
                f"mean_rollout_length={planning_stats['group_status/mean_rollout_length']:.1f}, "
                f"truncated={planning_stats['group_status/truncated_rollouts']}, "
                f"mean_ptc={planning_stats['group_status/mean_ptc_count']:.2f}, "
                f"mean_ptc_err={planning_stats['group_status/mean_ptc_error_count']:.2f}, "
                f"mean_ptc_penalty={planning_stats['reward/mean_ptc_error_penalty']:.4f}, "
                f"groups={planning_stats['group_status/total_groups']} "
                f"(all_zero={planning_stats['group_status/all_zero_ratio']:.2%}, "
                f"all_one={planning_stats['group_status/all_one_ratio']:.2%})"
            )

        if (
            not validate
            and self.dump_experience_every > 0
            and self.step_count % self.dump_experience_every == 0
        ):
            self._dump_experience(
                output=output,
                raw_prompts=raw_prompts[:batch_size],
                task_uids=task_uids[:batch_size],
                episode_rewards=episode_rewards,
                episode_successes=episode_successes,
                episode_turns=episode_turns,
            )

        return output

    # ------------------------------------------------------------------
    # SeparateRayPPOTrainer hooks
    # ------------------------------------------------------------------
    def _fit_compute_reward(self, batch: DataProto) -> DataProto:
        """TaskSync sets ``token_level_scores`` directly during rollout, so
        we just promote it to ``self.reward_tensor`` and skip extract_reward."""
        self.reward_tensor = batch.batch["token_level_scores"]
        self.reward_extra_infos_dict = {}
        return batch

    def _fit_validate(self):
        """Override: TaskSync's _validate uses a sync rollout path."""
        if self.val_planning_dataset is None:
            return
        if self.config.trainer.test_freq <= 0:
            return
        if not (self.is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
            return
        with marked_timer("testing", self.timing_raw, color="green"):
            val_metrics: dict = self._validate()
            if self.is_last_step:
                self.last_val_metrics = val_metrics
        self.metrics.update(val_metrics)

    def _fit_save_checkpoint(self):
        if self.config.trainer.save_freq <= 0:
            return
        if not (self.is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
            return
        with marked_timer("save_checkpoint", self.timing_raw, color="green"):
            self._save_checkpoint()

    def _fit_collect_metrics(self, batch):
        super()._fit_collect_metrics(batch)
        # Surface planning_stats from rollout.
        planning_stats = batch.meta_info.get("planning_stats") if batch is not None else None
        if planning_stats:
            self.metrics.update(planning_stats)

        # One-step-off diagnostics:
        #   wait_prev_gen — the await on prev rollout future, already wrapped
        #     in time/gen by the parent. Alias under the README's name so the
        #     dashboard matches the published table verbatim.
        #   one_step_off/* — bottleneck ratios. If gen >> train, rollout is the
        #     ceiling (train fits inside gen's shadow). If train > gen_async,
        #     actor became the new ceiling — re-balance toward more rollout GPUs.
        timing = self.timing_raw or {}
        gen_wait = timing.get("gen")
        gen_async = timing.get("generate_async")
        update_actor = timing.get("update_actor")
        old_log_prob = timing.get("old_log_prob") or 0.0
        if gen_wait is not None:
            self.metrics["time/wait_prev_gen"] = gen_wait
        if gen_async is not None and update_actor is not None:
            train_total = update_actor + old_log_prob
            self.metrics["one_step_off/gen_to_train_ratio"] = (
                gen_async / max(train_total, 1e-6)
            )
            # Fraction of step time spent waiting on rollout. Near 0 = perfect
            # overlap; near 1 = rollout is the ceiling and we're just waiting.
            step_total = timing.get("step", gen_async + train_total)
            self.metrics["one_step_off/wait_fraction"] = (
                (gen_wait or 0.0) / max(step_total, 1e-6)
            )

    # ------------------------------------------------------------------
    # Validation — sync, mirrors TaskSyncTrainer._validate verbatim except
    # the rollout dispatch goes through the (sync wrapper) AgentLoopManager.
    # ------------------------------------------------------------------
    def _validate(self) -> dict:
        if self.val_planning_dataset is None:
            return {}

        dense_rewards: list[float] = []
        episode_rewards: list[float] = []
        episode_successes: list[bool] = []
        episode_turns: list[int] = []
        rollout_lengths: list[int] = []
        truncated_flags: list[int] = []
        ptc_counts: list[int] = []
        ptc_error_counts: list[int] = []
        ptc_error_penalties: list[float] = []
        python_counts: list[int] = []
        terminal_counts: list[int] = []
        env_tool_counts: list[int] = []
        uid2rewards: dict = defaultdict(list)

        for batch_idx in range(len(self.val_planning_dataset)):
            # @auto_await on _build_and_generate_async dispatches to the right
            # execution mode whether or not the surrounding loop is running.
            output = self._build_and_generate_async(
                dataset=self.val_planning_dataset,
                batch_index=batch_idx,
                validate=True,
            )
            if output is None or output.non_tensor_batch is None:
                continue
            ntb = output.non_tensor_batch

            def _as_list(key: str) -> list:
                arr = ntb.get(key)
                if arr is None:
                    return []
                return arr.tolist() if hasattr(arr, "tolist") else list(arr)

            dense_rewards.extend(_as_list("dense_reward"))
            episode_rewards.extend(_as_list("episode_reward"))
            episode_successes.extend(_as_list("episode_success"))
            episode_turns.extend(_as_list("episode_turns"))
            rollout_lengths.extend(_as_list("rollout_length"))
            truncated_flags.extend(_as_list("truncated"))
            ptc_counts.extend(_as_list("programmatic_tool_call_count"))
            ptc_error_counts.extend(_as_list("programmatic_tool_call_error_count"))
            ptc_error_penalties.extend(_as_list("ptc_error_penalty"))
            python_counts.extend(_as_list("execute_python_count"))
            terminal_counts.extend(_as_list("terminal_count"))
            env_tool_counts.extend(_as_list("env_tool_count"))

            uids = _as_list("uid")
            if uids:
                if "reward" in ntb:
                    sample_rewards = _as_list("reward")
                elif output.batch is not None and "token_level_rewards" in output.batch:
                    sample_rewards = output.batch["token_level_rewards"].sum(dim=-1).tolist()
                else:
                    sample_rewards = episode_rewards[-len(uids):]
                for uid, r in zip(uids, sample_rewards, strict=False):
                    uid2rewards[uid].append(r)

        if not dense_rewards:
            return {}

        reward_metrics = {
            "val/reward/score": float(np.mean(dense_rewards)),
            "val/reward/score_min": float(np.min(dense_rewards)),
            "val/reward/score_max": float(np.max(dense_rewards)),
            "val/reward/zero_reward_ratio": (
                dense_rewards.count(0) / len(dense_rewards) if dense_rewards else 1.0
            ),
            "val/reward/episode_reward_mean": (
                float(np.mean(episode_rewards)) if episode_rewards else 0.0
            ),
            "val/reward/success_rate": (
                float(np.mean(episode_successes)) if episode_successes else 0.0
            ),
        }

        total_groups = len(uid2rewards)
        all_zero_groups = sum(1 for vals in uid2rewards.values() if all(v == 0 for v in vals))
        all_one_groups = sum(1 for vals in uid2rewards.values() if all(v == 1 for v in vals))
        all_zero_ratio = all_zero_groups / total_groups if total_groups else 0.0
        all_one_ratio = all_one_groups / total_groups if total_groups else 0.0

        group_metrics = {
            "val/group_status/mean_episode_len": (
                float(np.mean(episode_turns)) if episode_turns else 0.0
            ),
            "val/group_status/mean_rollout_length": (
                float(np.mean(rollout_lengths)) if rollout_lengths else 0.0
            ),
            "val/group_status/truncated_rollouts": (
                int(sum(truncated_flags)) if truncated_flags else 0
            ),
            "val/group_status/total_groups": total_groups,
            "val/group_status/all_zero_groups": all_zero_groups,
            "val/group_status/all_zero_ratio": all_zero_ratio,
            "val/group_status/all_one_groups": all_one_groups,
            "val/group_status/all_one_ratio": all_one_ratio,
            "val/group_status/mean_ptc_count": (
                float(np.mean(ptc_counts)) if ptc_counts else 0.0
            ),
            "val/group_status/mean_ptc_error_count": (
                float(np.mean(ptc_error_counts)) if ptc_error_counts else 0.0
            ),
            "val/group_status/mean_python_count": (
                float(np.mean(python_counts)) if python_counts else 0.0
            ),
            "val/group_status/mean_terminal_count": (
                float(np.mean(terminal_counts)) if terminal_counts else 0.0
            ),
            "val/group_status/mean_env_tool_count": (
                float(np.mean(env_tool_counts)) if env_tool_counts else 0.0
            ),
            "val/reward/mean_ptc_error_penalty": (
                float(np.mean(ptc_error_penalties)) if ptc_error_penalties else 0.0
            ),
        }
        return {**reward_metrics, **group_metrics}

    def _create_empty_batch(self) -> DataProto:
        prompt_length = self.config.actor_rollout_ref.rollout.get("prompt_length", 8192)
        response_length = self.config.actor_rollout_ref.rollout.get("response_length", 4096)
        total_length = prompt_length + response_length
        batch = TensorDict(
            {
                "input_ids": torch.zeros((0, total_length), dtype=torch.long),
                "prompts": torch.zeros((0, prompt_length), dtype=torch.long),
                "responses": torch.zeros((0, response_length), dtype=torch.long),
                "attention_mask": torch.zeros((0, total_length), dtype=torch.long),
                "position_ids": torch.zeros((0, total_length), dtype=torch.long),
                "response_mask": torch.zeros((0, response_length), dtype=torch.long),
                "token_level_scores": torch.zeros((0, response_length), dtype=torch.float32),
                "token_level_rewards": torch.zeros((0, response_length), dtype=torch.float32),
            },
            batch_size=torch.Size([0]),
        )
        return DataProto(
            batch=batch,
            non_tensor_batch={"uid": np.array([], dtype=object)},
            meta_info={},
        )

    # ------------------------------------------------------------------
    # Checkpointing — same shape as TaskSyncTrainer's, but uses actor_wg
    # (the OneStepOff actor-only worker group).
    # ------------------------------------------------------------------
    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{self.global_steps}",
                "actor",
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get(
            "remove_previous_ckpt_in_save", False
        )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None)
            if not remove_previous_ckpt_in_save
            else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )

        local_mkdir_safe(local_global_step_folder)
        training_state = {
            "epoch": getattr(self, "_current_epoch", 0),
            "batch_idx": getattr(self, "_current_batch_idx", 0),
            "dataset_state": self.planning_dataset.state_dict(),
        }
        torch.save(training_state, os.path.join(local_global_step_folder, "training_state.pt"))
        logger.info(
            f"Saved training state: epoch={training_state['epoch']}, "
            f"batch_idx={training_state['batch_idx']}, global_steps={self.global_steps}"
        )

        actor_ckpt_cfg = self.config.actor_rollout_ref.actor.checkpoint
        async_save = (
            (hasattr(actor_ckpt_cfg, "async_save") and actor_ckpt_cfg.async_save)
            or ("async_save" in actor_ckpt_cfg and actor_ckpt_cfg["async_save"])
        )
        if async_save:
            return

        with open(
            os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"),
            "w",
        ) as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

        self._resume_epoch = 0
        self._resume_batch_idx = 0
        self._current_epoch = 0
        self._current_batch_idx = 0

        if self.config.trainer.resume_mode == "disable":
            return 0
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                logger.info("Training from scratch")
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str)
            assert "global_step_" in self.config.trainer.resume_from_path
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)

        logger.info(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        actor_path = os.path.join(global_step_folder, "actor")
        self.actor_rollout_wg.load_checkpoint(
            actor_path,
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        training_state_path = os.path.join(global_step_folder, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, weights_only=False)
            self._resume_epoch = training_state["epoch"]
            self._resume_batch_idx = training_state["batch_idx"] + 1
            if "dataset_state" in training_state:
                self.planning_dataset.load_state_dict(training_state["dataset_state"])
            logger.info(
                f"Restored training state: epoch={self._resume_epoch}, "
                f"batch_idx={self._resume_batch_idx}, global_steps={self.global_steps}"
            )
        else:
            steps_per_epoch = len(self.planning_dataset)
            self._resume_epoch = self.global_steps // max(1, steps_per_epoch)
            self._resume_batch_idx = self.global_steps % max(1, steps_per_epoch)
            logger.warning(
                f"No training_state.pt found, estimating resume position: "
                f"epoch={self._resume_epoch}, batch_idx={self._resume_batch_idx}"
            )

    # ------------------------------------------------------------------
    # Experience dumping — verbatim port from TaskSyncTrainer.
    # ------------------------------------------------------------------
    def _dump_experience(
        self,
        output: DataProto,
        raw_prompts: list[dict],
        task_uids: list[str],
        episode_rewards: list[float],
        episode_successes: list[bool],
        episode_turns: list[int],
    ):
        try:
            experiences: dict = {}
            batch_size = len(raw_prompts)
            ntb = output.non_tensor_batch or {}

            messages_arr = ntb.get("messages", None)
            tool_rewards_arr = ntb.get("tool_rewards", None)
            reward_extra_infos = (
                list(ntb["reward_extra_info"]) if "reward_extra_info" in ntb else []
            )

            for i in range(batch_size):
                task_info = raw_prompts[i].get("task_info", {})
                task_name = task_info.get("task_name", f"unknown_{i}")
                episode_key = f"episode_{i}_{task_name}"

                episode_data = {
                    "task_name": task_name,
                    "task_uid": task_uids[i] if i < len(task_uids) else None,
                    "reward": episode_rewards[i] if i < len(episode_rewards) else 0.0,
                    "success": (
                        bool(episode_successes[i]) if i < len(episode_successes) else False
                    ),
                    "num_turns": episode_turns[i] if i < len(episode_turns) else 0,
                }

                if (
                    messages_arr is not None
                    and i < len(messages_arr)
                    and messages_arr[i] is not None
                ):
                    episode_data["messages"] = (
                        list(messages_arr[i])
                        if hasattr(messages_arr[i], "__iter__")
                        else messages_arr[i]
                    )

                if (
                    tool_rewards_arr is not None
                    and i < len(tool_rewards_arr)
                    and tool_rewards_arr[i] is not None
                ):
                    episode_data["tool_rewards"] = (
                        list(tool_rewards_arr[i])
                        if hasattr(tool_rewards_arr[i], "__iter__")
                        else tool_rewards_arr[i]
                    )

                if "messages" not in episode_data and i < len(reward_extra_infos):
                    extra = reward_extra_infos[i]
                    if isinstance(extra, dict):
                        if "messages" in extra:
                            episode_data["messages"] = extra["messages"]
                        if "tool_rewards" in extra:
                            episode_data["tool_rewards"] = extra["tool_rewards"]

                experiences[episode_key] = episode_data

            dump_path = os.path.join(
                self.game_state_save_path,
                f"actor{self.actor_id}_step{self.step_count}.json",
            )
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(experiences, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"Dumped {len(experiences)} experiences to {dump_path}")
        except Exception as e:
            logger.warning(f"Failed to dump experience: {e}")
