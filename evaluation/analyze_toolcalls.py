#!/usr/bin/env python3
import argparse
import os
import json
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ────────────────────────────────────────────────────────────────────────────────
# Shared config (same args pattern as your other scripts)
# ────────────────────────────────────────────────────────────────────────────────

_AGENT_DISPLAY = {
    "replan_low_level_gt": "Interleaving Exec.",
    "low_level_gt": "One-shot Exec.",
    "high_level_gt": "high-level",
}

def _pretty_agent(raw: str) -> str:
    for orig, nice in _AGENT_DISPLAY.items():
        if raw.startswith(orig):
            return f"{nice}{raw[len(orig):]}"
    return raw

def _tool_family(name: str) -> str:
    if name.startswith("robot_"):
        return "robot"
    if name.startswith("search_in_memory"):
        return "search"
    if name.startswith("inspect_"):
        return "inspect"
    if name == "pause_and_think":
        return "meta"
    if name == "terminate":
        return "terminate"
    return "other"

_FAMILY_PALETTES = {
    "robot":      plt.cm.Oranges,
    "search":     plt.cm.Blues,
    "inspect":    plt.cm.Purples,
    "meta":       plt.cm.Greens,
    "terminate":  plt.cm.Greys,
    "other":      plt.cm.tab20c,   # fallback for misc tools
}


# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Analyze reasoning tool-call logs.")
    p.add_argument("--data_dir", type=str, default="evaluation/sim_outputs/")
    p.add_argument("--output_dir", type=str, default="evaluation/sim_outputs/")
    p.add_argument("--task_config", type=str, default="evaluation/config/tasks_sim_all.txt")
    p.add_argument("--agent_types", nargs="+",
                   default=["low_level_gt", "replan_low_level_gt"])
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def _load_json_safe(path: str):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error in {path}: {e}")
    return None

def _split_variant(task_type: str) -> Tuple[str, str]:
    """Return (base_task_type, variant) where variant ∈ {'interactive','noninteractive'}."""
    if task_type.endswith("_interactive"):
        return task_type[:-12], "interactive"
    return task_type, "noninteractive"

def parse_task_config(task_config: str) -> Dict[str, List[str]]:
    """Parse config into {task_type: [uid, ...]}."""
    tasks: Dict[str, List[str]] = defaultdict(list)
    with open(task_config, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                task_type, fname = line.split("/", 1)
            except ValueError:
                print(f"[WARN] Cannot parse line: {line}")
                continue
            uid, ext = os.path.splitext(os.path.basename(fname))
            if ext != ".json":
                print(f"[WARN] Expected .json, got {ext} in: {line}")
            tasks[task_type].append(uid)
    return dict(tasks)


# ────────────────────────────────────────────────────────────────────────────────
# Parse results (ONE parser function)
# ────────────────────────────────────────────────────────────────────────────────

def parse_toolcall_results(args, tasks_dict: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Returns a row per tool-call:
      columns: [agent, task_type, base_task_type, variant, uid, tool_name]
    If a JSON file is missing or empty, it contributes no rows (not counted).
    """
    rows = []
    for task_type, uids in tasks_dict.items():
        base_task, variant = _split_variant(task_type)
        for uid in uids:
            for agent in args.agent_types:
                fname = f"reasoning_toolcalls_{agent}_{uid}.json"
                path = os.path.join(args.data_dir, task_type, uid, fname)
                data = _load_json_safe(path)
                if not data or not isinstance(data, list):
                    continue
                for item in data:
                    # accept entries that look like tool calls
                    if isinstance(item, dict) and item.get("type") == "tool_call":
                        name = item.get("name", "UNKNOWN")
                        rows.append({
                            "agent": agent,
                            "task_type": task_type,
                            "base_task_type": base_task,
                            "variant": variant,
                            "uid": uid,
                            "tool_name": name,
                        })
    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] No tool-call rows parsed. Check your paths/args.")
    return df


# ────────────────────────────────────────────────────────────────────────────────
# ANALYSIS 1: Per-agent tool distribution + pie/donut plots
# ────────────────────────────────────────────────────────────────────────────────

def plot_tool_distributions_by_agent(args, calls_df: pd.DataFrame, top_k: int = 10, pct_label_min: float = 3.0):
    """
    For each agent: donut chart of tool-call distribution with:
      - top_k tools shown explicitly, remaining grouped into 'Other'
      - consistent colors by tool *family* (robot/search/inspect/meta/terminate)
      - percentages only for slices ≥ pct_label_min
      - center label with total calls
    """
    os.makedirs(args.output_dir, exist_ok=True)
    if calls_df.empty:
        print("[WARN] No data to plot tool distributions.")
        return

    for agent in args.agent_types:
        sub = calls_df[calls_df["agent"] == agent]
        if sub.empty:
            print(f"[INFO] No calls for agent '{agent}', skipping pie.")
            continue

        counts = sub["tool_name"].value_counts()
        # Group long tail
        if len(counts) > top_k:
            top = counts.iloc[:top_k]
            other_sum = counts.iloc[top_k:].sum()
            if other_sum > 0:
                counts = pd.concat([top, pd.Series({"Other": int(other_sum)})])
            else:
                counts = top
        # (optional) keep integer dtype
        counts = counts.astype(int)

        total = int(counts.sum())
        if total == 0:
            print(f"[INFO] No tool calls for agent '{agent}', skipping pie.")
            continue

        # Build colors: shade within family so related tools look related
        names = counts.index.tolist()
        values = counts.values
        fam_lists: Dict[str, List[str]] = defaultdict(list)
        for nm in names:
            fam_lists[_tool_family(nm) if nm != "Other" else "other"].append(nm)

        # Precompute shade positions per family
        fam_shades = {fam: np.linspace(0.35, 0.85, len(nlist)) for fam, nlist in fam_lists.items()}

        colors = []
        fam_seen_idx: Dict[str, int] = defaultdict(int)
        for nm in names:
            if nm == "Other":
                colors.append((0.83, 0.83, 0.83, 1.0))  # gentle gray for Other
            else:
                fam = _tool_family(nm)
                palette = _FAMILY_PALETTES.get(fam, plt.cm.tab20c)
                idx = fam_seen_idx[fam]
                shade = fam_shades[fam][idx]
                colors.append(palette(shade))
                fam_seen_idx[fam] += 1

        # Percentage labels: only for sufficiently large slices
        def _autopct(pct):
            return f"{pct:.1f}%" if pct >= pct_label_min else ""

        fig = plt.figure(figsize=(9.5, 6.2))
        ax = fig.add_axes([0.07, 0.18, 0.60, 0.74])   # [left, bottom, width, height]

        wedges, texts, autotexts = ax.pie(
            values,
            labels=None,
            colors=colors,
            autopct=_autopct,
            pctdistance=0.70,       # slightly closer to the center
            startangle=90,
            counterclock=False,
            wedgeprops=dict(linewidth=1.2, edgecolor="white"),
            textprops=dict(color="black", fontsize=10)
        )

        # 2) Donut thickness + center text
        centre_circle = plt.Circle((0, 0), 0.50, fc="white")  # was 0.58 → thicker ring
        ax.add_artist(centre_circle)
        ax.text(0, 0, f"n={total}", ha="center", va="center", fontsize=13, weight="bold")
        ax.axis("equal")

        # 3) Title & legend: move legend to bottom, two columns
        fig.suptitle(
            f"Tool-call distribution — { _pretty_agent(agent) }   (n={total})",
            x=0.06, ha="left", fontsize=14
        )

        pct = (counts / total * 100).round(1)
        legend_labels = [f"{nm} ({cnt}, {p}%)" for nm, cnt, p in zip(names, values, pct)]

        fig.legend(
            wedges, legend_labels,
            title="Tool (count, %)",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=2,                 # <= multi-column legend
            frameon=False,
            borderpad=0.2, handlelength=1.2, handletextpad=0.6, columnspacing=1.0
        )

        fig.subplots_adjust(top=0.86, bottom=0.16, left=0.06, right=0.98)

        out = os.path.join(args.output_dir, f"toolcalls_pie_{agent}.png")
        plt.tight_layout()
        plt.savefig(out, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"[INFO] Saved pie: {out}")


# ────────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2: Average TOTAL tool calls per task for each agent (print)
# ────────────────────────────────────────────────────────────────────────────────

def print_avg_total_calls_per_agent(calls_df: pd.DataFrame):
    """
    For each agent, compute mean number of tool calls per (task run), where a run is (task_type, uid).
    """
    if calls_df.empty:
        print("[WARN] No tool-call data for averages.")
        return

    # count per run
    per_run = (
        calls_df
        .groupby(["agent", "task_type", "uid"], as_index=False)
        .size()
        .rename(columns={"size": "n_calls"})
    )
    # average per agent
    avg = per_run.groupby("agent")["n_calls"].mean().reset_index()
    print("\n=== Average total tool calls per task (by agent) ===")
    for _, r in avg.iterrows():
        print(f"{r['agent']:20s}  avg calls/run: {r['n_calls']:.2f}")


# ────────────────────────────────────────────────────────────────────────────────
# ANALYSIS 3: Average 'navigate' calls for INTERACTIVE tasks (print)
# ────────────────────────────────────────────────────────────────────────────────

def print_avg_navigate_replan(calls_df: pd.DataFrame):
    """
    For agents whose name contains 'replan', compute average # of 'navigate'
    tool calls per run (task_type, uid) and print:
      - overall average
      - interactive-only average
      - non-interactive-only average

    A 'run' here is any (agent, task_type, uid, variant) that appears in calls_df.
    Runs with zero 'navigate' calls are included with count=0.
    """
    if calls_df.empty:
        print("[WARN] No tool-call data for 'replan' navigate averages.")
        return

    # keep only 'replan' agents
    mask_replan = calls_df["agent"].str.contains("replan", na=False)
    rep = calls_df[mask_replan]
    if rep.empty:
        print("\n[INFO] No agents with 'replan' in name — skipping navigate averages.")
        return

    # Universe of runs (so zero-navigate runs are kept)
    runs = rep[["agent", "task_type", "uid", "variant"]].drop_duplicates()

    # Count navigate calls per run
    nav = (
        rep[rep["tool_name"] == "robot_navigate"]
        .groupby(["agent", "task_type", "uid", "variant"], as_index=False)
        .size()
        .rename(columns={"size": "n_navigate"})
    )

    # Merge counts onto all runs (fill missing with 0)
    merged = runs.merge(nav, on=["agent", "task_type", "uid", "variant"], how="left")
    merged["n_navigate"] = merged["n_navigate"].fillna(0)

    # Helper to print a group safely
    def _safe_mean(df, label):
        if df.empty:
            return "n/a"
        return f"{df['n_navigate'].mean():.2f}"

    print("\n=== Average 'navigate' calls per run — replan agents ===")
    for agent, g in merged.groupby("agent"):
        overall = _safe_mean(g, "overall")
        inter   = _safe_mean(g[g["variant"] == "interactive"], "interactive")
        nonint  = _safe_mean(g[g["variant"] == "noninteractive"], "non-interactive")
        print(f"{agent:24s}  overall: {overall:>5}   interactive: {inter:>5}   non-interactive: {nonint:>5}")

def print_avg_robot_skill_calls_replan(calls_df: pd.DataFrame):
    """
    For agents whose name contains 'replan', compute average # of 'robot_*'
    tool calls per run (agent, task_type, uid, variant).
    Prints overall, interactive-only, and non-interactive-only averages.
    Runs with zero robot calls are included with count=0.
    """
    if calls_df.empty:
        print("[WARN] No tool-call data for 'replan' robot-skill averages.")
        return

    # keep only 'replan' agents
    mask_replan = calls_df["agent"].str.contains("replan", na=False)
    rep = calls_df[mask_replan]
    if rep.empty:
        print("\n[INFO] No agents with 'replan' in name — skipping robot-skill averages.")
        return

    # Universe of runs (so zero-robot runs are kept)
    runs = rep[["agent", "task_type", "uid", "variant"]].drop_duplicates()

    # Count robot_* calls per run
    robot_mask = rep["tool_name"].str.startswith("robot_", na=False)
    robot = (
        rep[robot_mask]
        .groupby(["agent", "task_type", "uid", "variant"], as_index=False)
        .size()
        .rename(columns={"size": "n_robot"})
    )

    # Merge counts onto all runs (fill missing with 0)
    merged = runs.merge(robot, on=["agent", "task_type", "uid", "variant"], how="left")
    merged["n_robot"] = merged["n_robot"].fillna(0)

    def _fmt_mean(df):
        if df.empty:
            return "n/a"
        return f"{df['n_robot'].mean():.2f}"

    print("\n=== Average 'robot_*' calls per run — replan agents ===")
    for agent, g in merged.groupby("agent"):
        overall = _fmt_mean(g)
        inter   = _fmt_mean(g[g["variant"] == "interactive"])
        nonint  = _fmt_mean(g[g["variant"] == "noninteractive"])
        print(f"{agent:24s}  overall: {overall:>5}   interactive: {inter:>5}   non-interactive: {nonint:>5}")

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    tasks = parse_task_config(args.task_config)
    print(f"[INFO] Loaded {sum(len(v) for v in tasks.values())} tasks from config "
          f"across {len(tasks)} task types.")

    calls_df = parse_toolcall_results(args, tasks)
    # Three analyses:
    plot_tool_distributions_by_agent(args, calls_df)
    print_avg_total_calls_per_agent(calls_df)
    print_avg_navigate_replan(calls_df)
    print_avg_robot_skill_calls_replan(calls_df)

if __name__ == "__main__":
    main()
