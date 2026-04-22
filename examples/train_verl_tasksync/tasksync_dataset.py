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

"""Dataset for task-sync style environments.

task-sync envs have the structure:
    task_root/
        category_or_batch/
            task_name/
                v0/
                    data.py
                    env.py
                    prompt.md
                    task_verification.py

This module discovers and validates these envs, producing task_info dicts
compatible with TaskSyncAgentLoop. It also supports splitting discovered
tasks into train / val partitions deterministically.
"""

import logging
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict

logger = logging.getLogger(__name__)

_REQUIRED_FILES = ["data.py", "env.py", "prompt.md"]


class TaskSyncTaskInfo(TypedDict, total=False):
    source_path: str
    task_name: str


@dataclass
class TaskSyncEnvGroupBuilder:
    task_info: TaskSyncTaskInfo
    num_envs: int
    max_tool_calls: int = 50


class TaskSyncDataset:
    """Dataset that discovers and samples task-sync environments.

    Scans a directory tree for subdirectories containing the required files
    (data.py, env.py, prompt.md) and optionally validates them by running
    ``python data.py --db_path data.db --workspace workspace`` to ensure
    the data generation step works.

    Each batch contains ``batch_size`` different tasks with ``group_size``
    rollouts per task for GRPO reward centering.

    If ``split="train"`` / ``"val"`` is provided together with ``val_ratio``,
    the discovered tasks are deterministically partitioned: a seeded shuffle
    picks the first ``round(len(tasks) * val_ratio)`` entries as the val set
    and the remainder as train. Pass ``split=None`` to disable partitioning.
    """

    def __init__(
        self,
        tasks_dir: str,
        batch_size: int,
        group_size: int,
        max_tool_calls: int = 50,
        filter_pattern: Optional[str] = None,
        seed: int = 0,
        validate_on_discovery: bool = False,
        split: Optional[str] = None,
        val_ratio: float = 0.05,
    ):
        self.tasks_dir = tasks_dir
        self.batch_size = batch_size
        self.group_size = group_size
        self.max_tool_calls = max_tool_calls
        self.filter_pattern = filter_pattern
        self.seed = seed
        self.validate_on_discovery = validate_on_discovery
        self.split = split
        self.val_ratio = val_ratio

        all_tasks = self._discover_tasks()
        self.tasks = self._partition_tasks(all_tasks)
        if not self.tasks:
            raise ValueError(
                f"No valid task-sync tasks found in {tasks_dir} for split={split!r}"
            )

        logger.info(
            f"TaskSyncDataset: split={split!r} discovered={len(all_tasks)} kept={len(self.tasks)}"
        )

        self.rng = random.Random(seed)
        self.shuffled_tasks = self.tasks.copy()
        self.rng.shuffle(self.shuffled_tasks)
        self.current_index = 0

    def _discover_tasks(self) -> list[TaskSyncTaskInfo]:
        import fnmatch

        base = Path(self.tasks_dir).resolve()
        if not base.exists():
            raise ValueError(f"tasks_dir does not exist: {self.tasks_dir}")

        tasks: list[TaskSyncTaskInfo] = []
        failed: list[str] = []

        for root, dirs, files in os.walk(str(base)):
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            if all(f in files for f in _REQUIRED_FILES):
                task_dir = root
                rel = os.path.relpath(task_dir, str(base))
                task_name = rel.replace(os.sep, "/")

                if self.filter_pattern and not fnmatch.fnmatch(task_name, self.filter_pattern):
                    continue

                task_info: TaskSyncTaskInfo = {
                    "source_path": task_dir,
                    "task_name": task_name,
                }

                if self.validate_on_discovery:
                    if self._validate_task(task_info):
                        tasks.append(task_info)
                    else:
                        failed.append(task_name)
                else:
                    tasks.append(task_info)

                dirs.clear()

        if failed:
            logger.warning(
                f"Filtered out {len(failed)} invalid task-sync tasks. "
                f"Examples: {failed[:5]}{'...' if len(failed) > 5 else ''}"
            )

        return tasks

    def _partition_tasks(self, all_tasks: list[TaskSyncTaskInfo]) -> list[TaskSyncTaskInfo]:
        if self.split is None or self.val_ratio <= 0:
            return all_tasks

        ordered = sorted(all_tasks, key=lambda t: t["task_name"])
        rng = random.Random(self.seed)
        shuffled = ordered.copy()
        rng.shuffle(shuffled)

        n_val = max(1, round(len(shuffled) * self.val_ratio)) if shuffled else 0
        val_set = shuffled[:n_val]
        train_set = shuffled[n_val:]

        if self.split == "train":
            return train_set
        if self.split == "val":
            return val_set
        raise ValueError(f"split must be one of 'train', 'val', None. Got: {self.split!r}")

    def _validate_task(self, task_info: TaskSyncTaskInfo) -> bool:
        import shutil
        import tempfile

        source_path = task_info["source_path"]
        tmp = tempfile.mkdtemp(prefix="tasksync_validate_")
        try:
            for f in _REQUIRED_FILES + ["task_verification.py"]:
                src = os.path.join(source_path, f)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(tmp, f))

            r = subprocess.run(
                [sys.executable, "data.py", "--db_path", "data.db", "--workspace", "workspace"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode != 0:
                logger.debug(
                    f"Validation failed for {task_info['task_name']}: {r.stderr[:200]}"
                )
                return False

            return os.path.exists(os.path.join(tmp, "data.db"))

        except Exception as e:
            logger.debug(f"Validation error for {task_info['task_name']}: {e}")
            return False
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def __len__(self) -> int:
        return max(len(self.tasks) // self.batch_size, 1)

    def get_batch(self, index: int) -> list[TaskSyncEnvGroupBuilder]:
        start_idx = (index * self.batch_size) % len(self.shuffled_tasks)

        if start_idx + self.batch_size > len(self.shuffled_tasks):
            self.rng.shuffle(self.shuffled_tasks)
            start_idx = 0

        batch_tasks = self.shuffled_tasks[start_idx: start_idx + self.batch_size]

        return [
            TaskSyncEnvGroupBuilder(
                task_info=task,
                num_envs=self.group_size,
                max_tool_calls=self.max_tool_calls,
            )
            for task in batch_tasks
        ]

    def state_dict(self) -> dict:
        return {
            "rng_state": self.rng.getstate(),
            "shuffled_tasks": self.shuffled_tasks.copy(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.rng.setstate(state_dict["rng_state"])
        self.shuffled_tasks = state_dict["shuffled_tasks"]

    def get_task_by_name(self, task_name: str) -> Optional[TaskSyncTaskInfo]:
        for task in self.tasks:
            if task["task_name"] == task_name:
                return task
        return None
