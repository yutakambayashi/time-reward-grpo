"""Stack the evaluation videos of several policy versions side by side, one MP4 per scenario.

Usage: python build_compare.py <scenarios.json> <out_dir> <label>=<eval_run_dir> ...

Each eval run dir holds evaluation/update_*_episode_outcomes.json and videos/*/*.gif (one GIF per
scenario). A panel that finishes early holds its last frame, so the time difference stays visible.
Writes <out_dir>/media/<scenario>.mp4 and <out_dir>/manifest.json (success / T per version).
"""
import glob, json, os, subprocess, sys
from pathlib import Path

scen_file, out = sys.argv[1], Path(sys.argv[2])
runs = [arg.split("=", 1) for arg in sys.argv[3:]]
scenarios = json.load(open(scen_file))["scenarios"]
(out / "media").mkdir(parents=True, exist_ok=True)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
has_font = os.path.exists(FONT)


FPS = 30     # output frame rate
SPEED = 40   # simulator steps shown per second of video (LIBERO runs at 20 Hz, so 2x real time)


def frames(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip().split(",")[0])
    except ValueError:
        return 0


outcomes, videos = {}, {}
for label, run_dir in runs:
    files = sorted(glob.glob(f"{run_dir}/evaluation/update_*_episode_outcomes.json"))
    outcomes[label] = {ep["scenario_id"]: ep for ep in json.load(open(files[-1]))["episodes"]}
    videos[label] = glob.glob(f"{run_dir}/videos/*/*.gif")

manifest = {"labels": [label for label, _ in runs], "scenarios": []}
for sid in scenarios:
    key = sid.replace("/", "-")
    inputs, cells = [], []
    for label, _ in runs:
        ep = outcomes[label].get(sid, {})
        gif = next((p for p in videos[label] if p.endswith(f"{key}.gif")), None)
        cells.append({"label": label, "success": ep.get("success", 0) >= 1.0, "T": ep.get("env_steps"), "has_video": gif is not None})
        inputs.append(gif)
    entry = {"scenario": sid, "task": next(iter(outcomes.values())).get(sid, {}).get("task", ""), "cells": cells}
    if all(inputs):
        counts = [frames(p) for p in inputs]
        steps = [c["T"] or 280 for c in cells]
        entry["frames"] = counts
        longest = max(steps) / SPEED
        args, filters = ["ffmpeg", "-y", "-loglevel", "error"], []
        for i, (p, c) in enumerate(zip(inputs, cells)):
            args += ["-i", p]
            status = f"success  T={c['T']}" if c["success"] else "fail"
            text = ""
            if has_font:
                text = (f",drawtext=fontfile={FONT}:text='{c['label']}':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=4"
                        f",drawtext=fontfile={FONT}:text='{status}':x=8:y=h-30:fontsize=16:fontcolor={'0x7CFC9A' if c['success'] else '0xFF8080'}:box=1:boxcolor=black@0.55:boxborderw=4")
            # GIF frame delays are unreliable: place frame N at simulator step N*T/frames so every panel
            # shares one time axis in simulator steps, then play SPEED steps per second
            spf = steps[i] / max(counts[i], 1)
            filters.append(f"[{i}:v]setpts=N*{spf:.4f}/({SPEED}*TB),scale=-2:256,fps={FPS},"
                           f"tpad=stop_mode=clone:stop_duration={longest + 1:.2f}{text}[v{i}]")
        stack = "".join(f"[v{i}]" for i in range(len(inputs)))
        filters.append(f"{stack}hstack=inputs={len(inputs)}[out]")
        media = f"media/{key}.mp4"
        args += ["-filter_complex", ";".join(filters), "-map", "[out]", "-t", f"{longest + 1:.2f}",
                 "-pix_fmt", "yuv420p", "-movflags", "faststart", "-crf", "28", str(out / media)]
        subprocess.run(args, check=True)
        entry["media"] = media
    manifest["scenarios"].append(entry)
    print(sid, " | ".join(f"{c['label']} {'ok' if c['success'] else 'NG'} T={c['T']}" for c in cells), "| video" if all(inputs) else "| missing video")

json.dump(manifest, open(out / "manifest.json", "w"), indent=1, ensure_ascii=False)
print("wrote", out / "manifest.json", "| font", has_font)
