"""Add the time-dependent reward, step/timing metrics and a zero-variance group filter to ART-Embodied.

Usage: python patch_time_reward.py <ART-Embodied checkout at commit a37617eb>

reward.kwargs (empty kwargs keeps the original success-only reward):
  time_weight   lambda >= 0, weight of the speed term

R = 1[success] * (1 + lambda * (1 - T / T_max)), times reward.scale.
T is the number of simulator steps until success, T_max is rollout.max_episode_steps.

Set ART_GROUP_FILTER=zero_variance when training to drop only groups whose rewards are all equal.
"""
import sys
from pathlib import Path

repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
MARK = "time_reward_patch"


def patch(path, edits):
    p = repo / path
    s = p.read_text()
    if MARK in s:
        print("already patched", path)
        return
    for old, new in edits:
        if s.count(old) != 1:
            raise SystemExit(f"{path}: anchor found {s.count(old)} times: {old[:70]!r}")
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", path)


# --- settings.py: accept and validate reward.kwargs, provide the reward formula ---
SETTINGS_GUARD_OLD = (
    "        if config.reward.kwargs:\n"
    "            raise ValueError(\n"
    "                \"LIBERO positive-control components do not consume reward.kwargs; \"\n"
    "                \"remove them or provide a custom components factory\"\n"
    "            )\n"
)
SETTINGS_GUARD_NEW = "        time_reward_params(config)  # time_reward_patch: validates reward.kwargs\n"
SETTINGS_TAIL = '''

# time_reward_patch
_TIME_REWARD_KEYS = {"time_weight"}


def time_reward_params(config: Any) -> dict[str, float] | None:
    """Validated time-dependent reward parameters, or None for success-only reward."""

    kwargs = dict(config.reward.kwargs or {})
    if not kwargs:
        return None
    unknown = sorted(set(kwargs).difference(_TIME_REWARD_KEYS))
    if unknown:
        raise ValueError(f"Unsupported LIBERO reward.kwargs: {unknown}")
    params = {"time_weight": float(kwargs.get("time_weight", 0.0))}
    if params["time_weight"] < 0.0:
        raise ValueError("reward.kwargs.time_weight must be >= 0")
    return params


def time_dependent_reward(
    *,
    success: bool,
    env_steps: int,
    max_episode_steps: int,
    scale: float,
    params: Mapping[str, float] | None,
) -> float:
    """Any success (reward in [1, 1 + lambda)) ranks above any failure (reward 0)."""

    if params is None:
        return float(scale) if success else 0.0
    if not success:
        return 0.0
    t_max = float(max_episode_steps)
    return float(scale) * (1.0 + params["time_weight"] * (1.0 - min(env_steps, t_max) / t_max))
'''

patch("examples/embodied/libero/settings.py", [
    (SETTINGS_GUARD_OLD, SETTINGS_GUARD_NEW),
])
p = repo / "examples/embodied/libero/settings.py"
if "def time_reward_params" not in p.read_text():
    p.write_text(p.read_text() + SETTINGS_TAIL)

# --- rollout.py: reward, env_steps and timing ---
STEP_OLD = (
    "                (\n"
    "                    observation,\n"
    "                    reward,\n"
    "                    terminated_value,\n"
    "                    truncated_value,\n"
    "                    info,\n"
    "                ) = environments[env_index].step(native_action)\n"
)
STEP_NEW = (
    "                _t_env = time.perf_counter()\n"
    + STEP_OLD
    + "                timing[\"time_env_step_s\"] += time.perf_counter() - _t_env\n"
    "                env_steps[env_index] = int(info.get(\"env_steps\", env_steps[env_index]))\n"
)

patch("examples/embodied/libero/rollout.py", [
    ("import sys\n", "import sys\nimport time  # time_reward_patch\n"),
    ("from .settings import LiberoSettings\n",
     "from .settings import LiberoSettings, time_dependent_reward, time_reward_params\n"),
    ("    completed_steps = [0] * len(contexts)\n",
     "    completed_steps = [0] * len(contexts)\n"
     "    env_steps = [0] * len(contexts)\n"
     "    time_params = time_reward_params(config)\n"
     "    timing = {\"time_predict_s\": 0.0, \"time_env_step_s\": 0.0}\n"),
    ("            if policy_client is None:\n                native_actions = None\n",
     "            _t_predict = time.perf_counter()\n"
     "            if policy_client is None:\n                native_actions = None\n"),
    ("            if len(recorded_actions) != len(active_indices):\n",
     "            timing[\"time_predict_s\"] += time.perf_counter() - _t_predict\n"
     "            if len(recorded_actions) != len(active_indices):\n"),
    (STEP_OLD, STEP_NEW),
    ("                    float(config.reward.scale) if successes[index] else 0.0,\n",
     "                    time_dependent_reward(\n"
     "                        success=successes[index],\n"
     "                        env_steps=env_steps[index],\n"
     "                        max_episode_steps=int(config.rollout.max_episode_steps),\n"
     "                        scale=float(config.reward.scale),\n"
     "                        params=time_params,\n"
     "                    ),\n"),
    ("                    \"episode_steps\": completed_steps[index],\n",
     "                    \"episode_steps\": completed_steps[index],\n"
     "                    \"env_steps\": env_steps[index],\n"
     "                    \"time_predict_s\": timing[\"time_predict_s\"],\n"
     "                    \"time_env_step_s\": timing[\"time_env_step_s\"],\n"),
])

# --- evaluation.py: surface the new metrics in episode_outcomes.json ---
patch("src/art_embodied/evaluation.py", [
    ("                row[\"duration_seconds\"] = outcome.metrics.get(\"duration\")\n",
     "                row[\"duration_seconds\"] = outcome.metrics.get(\"duration\")\n"
     "                row.update({  # time_reward_patch\n"
     "                    key: outcome.metrics[key]\n"
     "                    for key in (\"env_steps\", \"time_predict_s\", \"time_env_step_s\")\n"
     "                    if key in outcome.metrics\n"
     "                })\n"),
])

# --- flow_sde.py: optional zero-variance group filter (ART_GROUP_FILTER=zero_variance) ---
# The stock filter keeps groups whose mean reward lies in [lower, upper] (0.1-0.9 in the examples).
# Time-dependent rewards exceed 1 for successes, so that range would drop all-success groups.
# With binary rewards both filters drop exactly the same groups (all 0 or all 1).
patch("src/art_embodied/backends/flow_sde.py", [
    ("import math\n", "import math\nimport os  # time_reward_patch\n"),
    ("        keep_group = not filter_rewards or _in_reward_filter(\n"
     "            group_mean,\n"
     "            lower=rewards_lower_bound,\n"
     "            upper=rewards_upper_bound,\n"
     "        )\n",
     "        if os.environ.get(\"ART_GROUP_FILTER\") == \"zero_variance\":\n"
     "            keep_group = not filter_rewards or float(reward_tensor[group_index].std()) > 1e-6\n"
     "        else:\n"
     "            keep_group = not filter_rewards or _in_reward_filter(\n"
     "                group_mean,\n"
     "                lower=rewards_lower_bound,\n"
     "                upper=rewards_upper_bound,\n"
     "            )\n"),
])
