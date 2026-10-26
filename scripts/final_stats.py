"""Numbers in the report: success counts, paired speed comparisons, per-task results, groups used for training.

Usage: python final_stats.py [results dir] > results/final_stats.txt

Reads <results>/eval/<version>.json (episode_outcomes.json of each 100-episode evaluation) and
<results>/reward_evidence.csv (one row per training trajectory, written by collect_rewards.py).
Speed is compared only on initial states that both versions solved; the sign test is two-sided
and ignores ties.
"""
import csv, json, math, statistics as st, sys
from collections import defaultdict
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results")
VERSIONS = {"SFT": "sft", "S10": "success_b10", "S20": "success_b20", "S40": "success_b40",
            "T20": "time_b20", "T40": "time_b40"}
ok = lambda ep: ep.get("success", 0) >= 1.0
D = {k: {ep["scenario_id"]: ep for ep in json.load(open(R / "eval" / f"{v}.json"))["episodes"]}
     for k, v in VERSIONS.items()}


def sign_p(a, b):
    n, k = a + b, min(a, b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


print("== success (/100) and median T of successful episodes")
for k, eps in D.items():
    T = [e["env_steps"] for e in eps.values() if ok(e)]
    print(f"{k} success {len(T)} median T {st.median(T):g}")


def paired(a, b):
    A, B = D[a], D[b]
    common = [s for s in A if ok(A[s]) and ok(B[s])]
    d = [B[s]["env_steps"] - A[s]["env_steps"] for s in common]
    fa, sl, eq = sum(x < 0 for x in d), sum(x > 0 for x in d), sum(x == 0 for x in d)
    ma = st.median(A[s]["env_steps"] for s in common)
    mb = st.median(B[s]["env_steps"] for s in common)
    onlyA = sum(ok(A[s]) and not ok(B[s]) for s in A)
    onlyB = sum(ok(B[s]) and not ok(A[s]) for s in A)
    print(f"{a}->{b} common {len(common)} faster/slower/same {fa}/{sl}/{eq} median diff {st.median(d):+g} "
          f"p {sign_p(fa, sl):.2g} | medians {ma:g} -> {mb:g} | success only-{a} {onlyA} only-{b} {onlyB} p {sign_p(onlyA, onlyB):.2g}")


print("\n== paired comparisons (B - A on initial states both solved)")
for a, b in [("S40", "T40"), ("SFT", "S10"), ("SFT", "S20"), ("SFT", "S40"), ("SFT", "T20"), ("SFT", "T40"), ("T20", "T40")]:
    paired(a, b)


def task_diff(a, b, ids):
    common = [s for s in ids if ok(D[a][s]) and ok(D[b][s])]
    if not common:
        return "n0"
    ma, mb = st.median(D[a][s]["env_steps"] for s in common), st.median(D[b][s]["env_steps"] for s in common)
    return f"n{len(common)} {ma:g}->{mb:g} ({mb - ma:+g})"


print("\n== per task: success SFT/S40/T40 (/10) | median T on initial states both solved")
for t in range(10):
    ids = [s for s in D["SFT"] if f"/task-{t:02d}/" in s]
    suc = "/".join(str(sum(ok(D[v][s]) for s in ids)) for v in ("SFT", "S40", "T40"))
    print(f"task {t} success {suc} | SFT->S40 {task_diff('SFT', 'S40', ids)} | SFT->T40 {task_diff('SFT', 'T40', ids)} "
          f"| S40->T40 {task_diff('S40', 'T40', ids)} | {D['SFT'][ids[0]].get('task', '')}")

print("\n== groups used for training per batch (4 groups of 8 trajectories; a group is used when its rewards differ)")
rows = list(csv.DictReader(open(R / "reward_evidence.csv")))
g = defaultdict(list)
for r in rows:
    g[(r["run"], int(r["update"]), r["group"])].append(float(r["reward"]))
for run in sorted({r["run"] for r in rows}):
    batches = sorted({u for rr, u, _ in g if rr == run})
    used = [sum(1 for (rr, u, _), v in g.items() if rr == run and u == b and st.pstdev(v) > 0) for b in batches]
    allok = [sum(1 for (rr, u, _), v in g.items() if rr == run and u == b and all(x >= 1.0 for x in v)) for b in batches]
    print(run, "used", sum(used), "/", 4 * len(batches), "series", "".join(map(str, used)))
    print(run, "all-success groups", sum(allok), "series", "".join(map(str, allok)))
