"""Collect the reward and success of every training trajectory into one CSV.

Usage: python collect_rewards.py <out.csv> <label>=<run output dir> ...
  e.g. python collect_rewards.py reward_evidence.csv S=$R/grpo-success T=$R/grpo-time

Reads <run>/rollout-evidence/policy-version-*.json written by ART-Embodied during training.
Columns: run, update (batch index from 0), group, task, signal_class, traj, reward, success.
"""
import csv, glob, json, os, sys

out = sys.argv[1]
rows = []
for arg in sys.argv[2:]:
    label, run_dir = arg.split("=", 1)
    files = sorted(glob.glob(f"{run_dir}/rollout-evidence/policy-version-*.json"))
    print(label, run_dir, len(files), "batches")
    for f in files:
        d = json.load(open(f))
        for g in d["groups"]:
            update = g["group_metadata"].get("update", d.get("policy_version"))
            task = g["scenario_id"].split("/")[3]
            for i, (r, s) in enumerate(zip(g["rewards"], g["successes"])):
                rows.append([label, update, g["group_index"], task, g["signal_class"], i, r, s])
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["run", "update", "group", "task", "signal_class", "traj", "reward", "success"])
    w.writerows(rows)
print("wrote", out, len(rows), "rows")
