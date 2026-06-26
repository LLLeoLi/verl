#!/bin/bash
# ==============================================================================
# Entry script for ONE-STEP-OFF-POLICY TaskSync training (verl 0.7.1).
#
# Usage (from the verl repo root):
#   NNODES=6 NGPUS_PER_NODE=8 bash aiic_recipe/train_megatron_one_step_off.sh \
#       [--actor_nnodes 4 --rollout_nnodes 2] \
#       [--exp_name ...] [--model_path ...] [...]
#
# Default split (NNODES=6): 32 GPU actor (4 nodes) + 16 GPU rollout (2 nodes).
# Total physical nodes required = actor_nnodes + rollout_nnodes when each
# pool fills its nodes (n_gpus_per_node = 8 on both sides).
# ==============================================================================
set -x

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ==============================================================================
# CLI argument parsing
# ==============================================================================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --exp_name) exp_name="$2"; shift 2 ;;
        --model_path) model_path="$2"; shift 2 ;;
        --dist_ckpt_path) dist_ckpt_path="$2"; shift 2 ;;
        --use_dist_checkpointing) use_dist_checkpointing="$2"; shift 2 ;;
        --data_dir) data_dir="$2"; shift 2 ;;
        --ckpt_root) ckpt_root="$2"; shift 2 ;;
        --prompt_len) prompt_len="$2"; shift 2 ;;
        --response_len) response_len="$2"; shift 2 ;;
        --env_batch_size) env_batch_size="$2"; shift 2 ;;
        --env_group_size) env_group_size="$2"; shift 2 ;;
        --ppo_micro_bsz_per_gpu) ppo_micro_bsz_per_gpu="$2"; shift 2 ;;
        --temperature) temperature="$2"; shift 2 ;;
        --clip_ratio_high) clip_ratio_high="$2"; shift 2 ;;
        --total_epochs) total_epochs="$2"; shift 2 ;;
        --loss_mode) loss_mode="$2"; shift 2 ;;
        --actor_lr) actor_lr="$2"; shift 2 ;;
        --lr_warmup_steps) lr_warmup_steps="$2"; shift 2 ;;
        --weight_decay) weight_decay="$2"; shift 2 ;;
        --rollout_tp) rollout_tp="$2"; shift 2 ;;
        --actor_tp) actor_tp="$2"; shift 2 ;;
        --actor_pp) actor_pp="$2"; shift 2 ;;
        --actor_cp) actor_cp="$2"; shift 2 ;;
        --actor_ep) actor_ep="$2"; shift 2 ;;
        --actor_etp) actor_etp="$2"; shift 2 ;;
        --offload) offload="$2"; shift 2 ;;
        --offload_fraction) offload_fraction="$2"; shift 2 ;;
        --use_mbridge) use_mbridge="$2"; shift 2 ;;
        --gpu_memory_utilization) gpu_memory_utilization="$2"; shift 2 ;;
        --max_token_len_per_gpu) max_token_len_per_gpu="$2"; shift 2 ;;
        --num_workers) num_workers="$2"; shift 2 ;;
        --save_freq) save_freq="$2"; shift 2 ;;
        --dump_experience_every) dump_experience_every="$2"; shift 2 ;;
        --max_tool_calls) max_tool_calls="$2"; shift 2 ;;
        --reward_type) reward_type="$2"; shift 2 ;;
        --ptc_mode) ptc_mode="$2"; shift 2 ;;
        --ptc_error_penalty) ptc_error_penalty="$2"; shift 2 ;;
        --dense_epoch) dense_epoch="$2"; shift 2 ;;
        --val_ratio) val_ratio="$2"; shift 2 ;;
        --test_freq) test_freq="$2"; shift 2 ;;
        --suffix) suffix="$2"; shift 2 ;;
        # one-step-off specific:
        --actor_nnodes) actor_nnodes="$2"; shift 2 ;;
        --actor_ngpus_per_node) actor_ngpus_per_node="$2"; shift 2 ;;
        --rollout_nnodes) rollout_nnodes="$2"; shift 2 ;;
        --rollout_ngpus_per_node) rollout_ngpus_per_node="$2"; shift 2 ;;
        --bypass_mode) bypass_mode="$2"; shift 2 ;;
        *) break ;;
    esac
done

# ==============================================================================
# Model / data paths
# ==============================================================================
model_path=${model_path:-/mnt/public_02/lihao/ptc-checkpoints/Qwen3-Coder-30B-A3B-ptc-SFT}
use_dist_checkpointing=${use_dist_checkpointing:-False}
dist_ckpt_path=${dist_ckpt_path:-null}
data_dir=${data_dir:-task-sync/claude-sync-v4/opus}

# ==============================================================================
# Sequence lengths
# ==============================================================================
prompt_len=${prompt_len:-5600}
response_len=${response_len:-25600}
total_len=$((prompt_len + response_len))

# ==============================================================================
# Core training params
# ==============================================================================
env_batch_size=${env_batch_size:-8}
env_group_size=${env_group_size:-32}
ppo_mini_bsz=$((env_batch_size * env_group_size))

ppo_micro_bsz_per_gpu=${ppo_micro_bsz_per_gpu:-4}
temperature=${temperature:-1.0}
clip_ratio_high=${clip_ratio_high:-0.28}
total_epochs=${total_epochs:-5}
loss_mode=${loss_mode:-gspo}
actor_lr=${actor_lr:-1e-6}
lr_warmup_steps=${lr_warmup_steps:-10}
weight_decay=${weight_decay:-0.1}

# ==============================================================================
# Actor parallelism
#   Default split (NNODES=6): actor=32 GPU, rollout=16 GPU
#   Recommended actor parallelism: TP=4 PP=2 CP=4 EP=8 (DP=1)
#     - EP=8 divides DP*TP*CP=16 ✓
#     - max_token_len_per_gpu auto-sized below to ceil(total_len/CP)
# ==============================================================================
actor_tp=${actor_tp:-4}
actor_pp=${actor_pp:-2}
actor_cp=${actor_cp:-2}
actor_ep=${actor_ep:-8}
actor_etp=${actor_etp:-1}

# Rollout parallelism
rollout_tp=${rollout_tp:-8}

# ==============================================================================
# Resource split (one-step-off)
#   Defaults: 4 nodes actor + 2 nodes rollout = 6 physical nodes total
#   Override via --actor_nnodes / --rollout_nnodes / --*_ngpus_per_node
# ==============================================================================
actor_nnodes=${actor_nnodes:-4}
actor_ngpus_per_node=${actor_ngpus_per_node:-${NGPUS_PER_NODE:-8}}
rollout_nnodes=${rollout_nnodes:-2}
rollout_ngpus_per_node=${rollout_ngpus_per_node:-${NGPUS_PER_NODE:-8}}

# ==============================================================================
# Dynamic-batch per-GPU token budget
# ==============================================================================
max_token_len_per_gpu=${max_token_len_per_gpu:-$(( (total_len + actor_cp - 1) / actor_cp ))}

# ==============================================================================
# Offload & misc
# ==============================================================================
offload=${offload:-True}
offload_fraction=${offload_fraction:-1.0}
use_mbridge=${use_mbridge:-True}
# rollout has dedicated GPUs now -- can be more aggressive than the hybrid default
gpu_memory_utilization=${gpu_memory_utilization:-0.85}
num_workers=${num_workers:-8}
save_freq=${save_freq:-25}
dump_experience_every=${dump_experience_every:-1}
max_tool_calls=${max_tool_calls:-100}
reward_type=${reward_type:-dense}
ptc_mode=${ptc_mode:-ptc}
case "${ptc_mode}" in
    ptc|no-ptc|mixed) ;;
    *) echo "ERROR: --ptc_mode must be 'ptc', 'no-ptc', or 'mixed' (got '${ptc_mode}')" >&2; exit 1 ;;
esac
ptc_error_penalty=${ptc_error_penalty:-0.00}
dense_epoch=${dense_epoch:-0}
val_ratio=${val_ratio:-0.0}
test_freq=${test_freq:--1}

# ==============================================================================
# Off-policy correction
#   bypass_mode=True : use rollout_log_probs as old_log_probs (cheaper but
#                     forces loss_mode="bypass_mode", IGNORING --loss_mode).
#   bypass_mode=False: decoupled mode, recompute old_log_prob normally and
#                     apply IS weights. Keeps your --loss_mode (e.g. gspo).
# ==============================================================================
bypass_mode=${bypass_mode:-True}

# ==============================================================================
# Experiment name
# ==============================================================================
DATE=$(date +%m%d)
dense_epoch_suffix=""
if [ "${dense_epoch}" -gt 0 ] 2>/dev/null; then
    dense_epoch_suffix="-dense_epoch${dense_epoch}"
fi
suffix=${suffix:-""}
suffix_str=""
if [ -n "${suffix}" ]; then
    suffix_str="-${suffix}"
fi
exp_name=${exp_name:-"${DATE}-tasksync-30b-a3b-1stepoff-${loss_mode}-tp${actor_tp}-pp${actor_pp}-cp${actor_cp}-ep${actor_ep}-etp${actor_etp}-bsz${ppo_micro_bsz_per_gpu}-total_epochs${total_epochs}-group_size${env_group_size}-reward_type${reward_type}${dense_epoch_suffix}${suffix_str}"}
ckpt_root=${ckpt_root:-"/mnt/public_02/lihao/ptc-checkpoints/${exp_name}"}

echo "${exp_name}"
echo "Resource split: actor=${actor_nnodes}x${actor_ngpus_per_node} rollout=${rollout_nnodes}x${rollout_ngpus_per_node}"

# ==============================================================================
# Build training command
# ==============================================================================
TRAIN_CMD=(
    python3 -m examples.train_verl_tasksync.train_tasksync_one_step_off
    --config-name=tasksync_grpo_megatron_one_step_off

    # model & data
    actor_rollout_ref.model.path=${model_path}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.env.tasks_dir=${data_dir}
    actor_rollout_ref.env.batch_size=${env_batch_size}
    actor_rollout_ref.env.group_size=${env_group_size}
    actor_rollout_ref.env.reward_type=${reward_type}
    actor_rollout_ref.env.max_tool_calls=${max_tool_calls}
    actor_rollout_ref.env.ptc_mode=${ptc_mode}
    actor_rollout_ref.env.ptc_error_penalty=${ptc_error_penalty}
    actor_rollout_ref.env.val_ratio=${val_ratio}

    # one-step-off resource split
    actor_rollout_ref.hybrid_engine=False
    rollout.nnodes=${rollout_nnodes}
    rollout.n_gpus_per_node=${rollout_ngpus_per_node}

    # rollout
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.prompt_length=${prompt_len}
    actor_rollout_ref.rollout.response_length=${response_len}
    actor_rollout_ref.rollout.max_model_len=${total_len}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.agent.default_agent_loop=tasksync_agent
    actor_rollout_ref.rollout.agent.num_workers=${num_workers}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ppo_micro_bsz_per_gpu}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_token_len_per_gpu}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    # MUST stay False for one-step-off (rollout engine kept alive across steps).
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
    actor_rollout_ref.dump_experience_every=${dump_experience_every}

    # actor training
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_bsz}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_bsz_per_gpu}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token_len_per_gpu}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.policy_loss.loss_mode="${loss_mode}"
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode="token-mean"
    actor_rollout_ref.actor.use_rollout_log_probs=True
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra"]'

    # actor megatron parallelism
    actor_rollout_ref.actor.megatron.use_mbridge=${use_mbridge}
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=${use_dist_checkpointing}
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${dist_ckpt_path}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${actor_cp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${actor_etp}

    # actor megatron fused kernels
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True

    # actor megatron MoE config
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    "+actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex"
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True

    # actor optimizer
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps}
    actor_rollout_ref.actor.optim.lr_decay_style='constant'
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay}
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${offload_fraction}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True

    # off-policy correction
    algorithm.rollout_correction.bypass_mode=${bypass_mode}

    # trainer
    trainer.logger='["console","wandb"]'
    trainer.project_name="gem-tasksync-training-verl"
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${actor_ngpus_per_node}
    trainer.nnodes=${actor_nnodes}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.dense_epoch=${dense_epoch}
    trainer.resume_mode=auto
    trainer.default_local_dir=${ckpt_root}

    # pass through any remaining args
    "$@"
)

"${TRAIN_CMD[@]}"
