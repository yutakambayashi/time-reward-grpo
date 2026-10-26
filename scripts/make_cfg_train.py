"""Write the GRPO config used in the report (one L4, LIBERO-Object, 4 groups x 8 trajectories per batch).

Usage: python make_cfg_train.py <ART-Embodied checkout> <out.yaml> <run> <batches> <eval_every> <eval_before:0|1> [reward_kwargs_json]

Empty reward_kwargs = success-only reward. '{"time_weight": 1.0}' = time-dependent reward (needs patch_time_reward.py).
Training must also run with ART_GROUP_FILTER=zero_variance so that only groups with equal rewards are dropped.
Outputs go to $ART_OUTPUT_ROOT/<run> (default /content/drive/MyDrive/art_embodied_runs/<run>).
Set ART_WANDB_PROJECT to also log metrics and rollout videos to W&B (needs `wandb login` first).
"""
import json, os, sys, yaml

repo, out, run = sys.argv[1], sys.argv[2], sys.argv[3]
batches, eval_every, eval_before = int(sys.argv[4]), int(sys.argv[5]), bool(int(sys.argv[6]))
reward_kwargs = json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}
GROUPS = 4  # rollout actors per batch, one 8-trajectory group each; 4 fits on an L4
ex = f"{repo}/examples/embodied"
c = yaml.safe_load(open(f"{ex}/smolvla_libero_object_flow_sde_grpo_positive_control_h100.yaml"))
smoke = yaml.safe_load(open(f"{ex}/smolvla_libero_object_flow_sde_grpo_smoke_h100.yaml"))
c["experiment"]["run"] = run
c["experiment"]["tags"] = list(c["experiment"].get("tags") or []) + ["group-filter-zero_variance"]

alg = c["algorithm"]
alg["flow_sde"]["noise_level"] = 0.5      # Flow-SDE noise (ART's K4 recipes)
alg["flow_sde"]["num_denoise_steps"] = 4  # also used by deterministic evaluation
alg["pre_update_logprob_kl_tolerance"] = 0.05  # rollout/rescore alignment guard (default 0.02)
alg["pre_update_ratio_tolerance"] = 0.05

c["policy"]["load_kwargs"]["execution_horizon"] = 10
env = c["environment"]["kwargs"]
env["action_chunk_size"] = 10
env["evaluation_protocol"] = "lerobot_v060_chunked"
env["init_state_selection"] = "partitioned_random_reset"
env["training_task_ids"] = list(range(10))
env["training_trial_ids"] = list(range(10, 50))  # evaluation uses init states 0-9
env["reset_schedule"]["total_num_processes"] = GROUPS
c["reward"]["kwargs"] = reward_kwargs
c["runtime"] = smoke["runtime"]
c["runtime"]["rollout_execution"]["actors_per_device"] = GROUPS
c["rollout"]["workers"] = GROUPS
c["rollout"]["groups_per_update"] = GROUPS * env["reset_schedule"]["groups_per_process_per_rollout_epoch"]
c["rollout"]["max_policy_steps"] = 28
c["rollout"]["max_episode_steps"] = 280
sch = c["training"]["schedule"]
sch["action_chunk_size"] = 10
sch["actor_world_size"] = 1
sch["approximate_kl_guard_scope"] = "primitive_action"
c["training"]["updates"] = batches

ev = c["evaluation"]
ev["evaluate_before_training"] = eval_before
ev["every_updates"] = eval_every
ob = c["observability"]
ob["wandb"]["enabled"] = False
ob["weave"]["enabled"] = False
ob["videos_per_evaluation"] = 1
ob["videos_per_update"] = 0
ob["require_train_video"] = False
ob["require_evaluation_video"] = False
wb_project = os.environ.get("ART_WANDB_PROJECT")
if wb_project:
    ob["delivery_failure_policy"] = "best_effort"
    ob["wandb"].update({"enabled": True, "entity": None, "project": wb_project, "group": run, "mode": "online", "log_model_artifacts": False})
    ob["videos_per_update"] = 2
    ob["videos_per_evaluation"] = 4

root = os.environ.get("ART_OUTPUT_ROOT", "/content/drive/MyDrive/art_embodied_runs")
st = c["storage"]
st["output_dir"] = f"{root}/{run}"
st["retain_checkpoint_updates"] = [u for u in (5, 10, 15, 20, 25, 30, 40, 50, 75, 100) if u <= batches]
yaml.safe_dump(c, open(out, "w"), sort_keys=False, allow_unicode=True)
print("wrote", out, "| batches", batches, "| trajectories/batch", GROUPS * c["algorithm"]["group_size"])
