"""Print success count and steps to success per task for one episode_outcomes.json.

Usage: python summarize.py <episode_outcomes.json>
"""
import collections, json, statistics as st, sys

eps = json.load(open(sys.argv[1]))["episodes"]
ok = lambda e: e["success"] >= 1.0
by_task = collections.defaultdict(list)
for e in eps:
    by_task[int(e["scenario_id"].split("task-")[1][:2])].append(e)
print(f"episodes {len(eps)} success {sum(ok(e) for e in eps)}")
for t, es in sorted(by_task.items()):
    T = sorted(e["env_steps"] for e in es if ok(e))
    print(f"task {t} success {len(T)}/{len(es)} T={T}")
T = [e["env_steps"] for e in eps if ok(e)]
print("median T", st.median(T), "min", min(T), "max", max(T))
