"""Pick evaluation scenarios that show the differences between policy versions side by side.

Usage: python select_scenarios.py <out.json> <label>=<episode_outcomes.json> ...

Per task, picks up to two scenarios:
  1. all versions succeed and the spread of T across versions is largest (shows speed),
  2. versions disagree on success (shows reliability); if none, the next-largest T spread.
"""
import json, sys
from collections import defaultdict

out = sys.argv[1]
runs = [arg.split("=", 1) for arg in sys.argv[2:]]
labels = [label for label, _ in runs]
data = {label: {ep["scenario_id"]: ep for ep in json.load(open(path))["episodes"]} for label, path in runs}
common = sorted(set.intersection(*(set(d) for d in data.values())))
ok = lambda ep: ep.get("success", 0) >= 1.0

by_task = defaultdict(list)
for sid in common:
    by_task[sid.split("/")[2]].append(sid)

picked = []
for task, sids in sorted(by_task.items()):
    all_ok = [s for s in sids if all(ok(data[l][s]) for l in labels)]
    all_ok.sort(key=lambda s: -(max(data[l][s]["env_steps"] for l in labels) - min(data[l][s]["env_steps"] for l in labels)))
    mixed = [s for s in sids if len({ok(data[l][s]) for l in labels}) > 1]
    # prefer disagreements where the last version (the proposed one) succeeds
    mixed.sort(key=lambda s: (not ok(data[labels[-1]][s]), -sum(ok(data[l][s]) for l in labels)))
    chosen = all_ok[:1] + (mixed[:1] if mixed else all_ok[1:2])
    for s in chosen:
        picked.append(s)
        row = " | ".join(f"{l} {'ok' if ok(data[l][s]) else 'NG'} T={data[l][s]['env_steps']}" for l in labels)
        print(s, "|", row)

json.dump({"labels": labels, "scenarios": picked}, open(out, "w"), indent=1)
print("picked", len(picked))
