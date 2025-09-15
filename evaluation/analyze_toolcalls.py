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
        return "Robot Skills"
    if name.startswith("search_in_memory"):
        return "Memory Search"
    if name.startswith("inspect_"):
        return "Memory Inspect"
    if name == "pause_and_think":
        return "Reflect"
    # if name == "terminate":
        # return "terminate"
    return "Other"

_FAMILY_PALETTES = {
    "Robot Skills":      plt.cm.Oranges,
    "Memory Search":     plt.cm.Blues,
    "Memory Inspect":    plt.cm.Purples,
    "Reflect":       plt.cm.Greens,
    # "terminate":  plt.cm.Greys,
    "Other":      plt.cm.Greys,  # fallback for misc tools
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
                   default=["low_level_gt", "replan_low_level_gt", "low_level_caption", "replan_low_level_caption"])
    p.add_argument("--family_only", action="store_true",
                   help="If set, aggregate tool calls by family in plots (robot/search/inspect/meta/terminate/other).")
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


def build_success_df(args, tasks_dict: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Load per-run success flags from results_{agent}_{uid}.json.
    Returns columns: [agent, task_type, uid, variant, success] where success ∈ {0,1}.
    Missing files => success=0.
    """
    rows = []
    for task_type, uids in tasks_dict.items():
        base_task, variant = _split_variant(task_type)
        for uid in uids:
            for agent in args.agent_types:
                # results file lives next to the tool-call logs
                fname = f"results_{agent}_{uid}.json"
                path  = os.path.join(args.data_dir, task_type, uid, fname)
                data  = _load_json_safe(path) or {}
                # fall back to 0 if missing/None
                succ  = int(bool(data.get("success", 0)))
                rows.append({
                    "agent": agent,
                    "task_type": task_type,
                    "uid": uid,
                    "variant": variant,
                    "success": succ,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] No success rows parsed. Check your paths/args.")
    return df


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

def plot_tool_distributions_by_agent(args, calls_df: pd.DataFrame, pct_label_min: float = 3.0):
    """
    Produce SIDE-BY-SIDE donut charts per env. pair with ONE legend per figure.

    Pairs plotted (if present in args.agent_types and data):
      - Realistic:  low_level_caption           vs replan_low_level_caption
      - Oracle:     low_level_gt                vs replan_low_level_gt

    Grouping:
      - Reflect                  := pause_and_think
      - Mem: Text Descsc Replay     := search_in_memory*
      - Mem: Image Replay        := inspect_*
      - Physical Actions         := robot_*
      - Other                    := else

    Colors:
      - Physical Actions: blue family
      - Memory (Txt/Image): orange family (two shades)
      - Reflect/Other: grays

    Legend (single, at figure-level): shows color → group with per-agent percentages, e.g.:
      "Physical Actions (TR+S: 54.3%, STAR: 32.1%)"
    """
    os.makedirs(args.output_dir, exist_ok=True)
    if calls_df.empty:
        print("[WARN] No data to plot tool distributions.")
        return

    # --- helpers ---------------------------------------------------------------
    def _pie_group_for_tool(name: str) -> str:
        if name == "pause_and_think":
            return "Reflect"
        if isinstance(name, str) and name.startswith("search_in_memory"):
            return "Mem: Text Descsc Replay"
        if isinstance(name, str) and name.startswith("inspect_"):
            return "Mem: Image Replay"
        if isinstance(name, str) and name.startswith("robot_"):
            return "Physical Actions"
        return "Other"

    def _method_env_from_agent(agent: str) -> tuple[str, str]:
        # Method
        if agent.startswith("replan_low_level"):
            method = "STAR"
        elif agent.startswith("low_level"):
            method = "TR+S"
        else:
            method = _pretty_agent(agent)
        # Env.
        if agent.endswith("_gt"):
            env = "(Oracle)"
        elif agent.endswith("_caption"):
            env = "(Realistic)"
        else:
            env = ""
        return method, env

    def _group_counts(df_agent: pd.DataFrame) -> tuple[pd.Index, np.ndarray, pd.Series]:
        grouped = df_agent["tool_name"].map(_pie_group_for_tool).value_counts()
        grouped = grouped.reindex([g for g in ORDER if g in grouped.index]).astype(int)
        names = grouped.index
        vals  = grouped.values
        total = max(int(np.sum(vals)), 1)
        pct   = (pd.Series(vals, index=names) / total * 100).round(1)
        return names, vals, pct

    # Fixed display order + colors
    ORDER = [
        "Physical Actions",
        "Mem: Text Descsc Replay",
        "Mem: Image Replay",
        "Reflect",
        "Other",
    ]
    COLORS = {
        "Physical Actions": plt.cm.Blues(0.65),
        "Mem: Text Descsc Replay": plt.cm.Oranges(0.55),
        "Mem: Image Replay": plt.cm.Oranges(0.80),
        "Reflect": (0.6, 0.6, 0.6, 1.0),
        "Other":   (0.83, 0.83, 0.83, 1.0),
    }

    # Define pairs to compare side-by-side
    pairs = [
        ("Realistic", ["low_level_caption", "replan_low_level_caption"]),
        ("Oracle",    ["low_level_gt",      "replan_low_level_gt"]),
    ]

    # Filter to requested agents
    valid_agents = set(args.agent_types)

    for env_label, agents_pair in pairs:
        a_left, a_right = agents_pair

        # Subsets present in args and data
        present = []
        for a in agents_pair:
            if a not in valid_agents:
                continue
            sub = calls_df[calls_df["agent"] == a]
            if not sub.empty:
                present.append(a)

        if len(present) < 2:
            # If we only have one, still skip (you asked for side-by-side).
            if len(present) == 1:
                print(f"[INFO] Only one agent with data for {env_label} ({present[0]}). Skipping paired plot.")
            else:
                print(f"[INFO] No agents with data for {env_label}. Skipping.")
            continue

        # Build figure with two subplots (left/right in fixed order)
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.8))  # narrower figure
        fig.subplots_adjust(
            top=0.86,
            bottom=0.20,
            left=0.06,
            right=0.98,
            wspace=-0.2   # reduce from 0.18 → 0.05
        )

        legend_labels = []   # combined legend entries per group
        legend_handles = []  # create once from left axis wedges' colors

        per_agent_pct = {}   # {group: {agent: pct}}

        method_names = {}    # agent -> "TR+S"/"STAR"
        env_names = {}       # agent -> "(Realistic)"/"(Oracle)"/""

        # We want fixed (left, right) = (a_left, a_right) if both present
        ordered_agents = [a_left, a_right]

        wedges_for_left = None  # to extract handles/colors

        for side_idx, agent in enumerate(ordered_agents):
            sub = calls_df[calls_df["agent"] == agent]
            names, vals, pct = _group_counts(sub)

            # Save percentages for combined legend
            method, env = _method_env_from_agent(agent)
            method_names[agent] = method
            env_names[agent] = env
            for g in names:
                per_agent_pct.setdefault(g, {})[agent] = pct[g]

            # Draw pie
            ax = axes[side_idx]
            def _autopct(p):
                return f"{p:.1f}%" if p >= pct_label_min else ""

            wedges, texts, autotexts = ax.pie(
                vals,
                labels=None,
                colors=[COLORS[g] for g in names],
                autopct=_autopct,
                pctdistance=0.70,
                startangle=90,
                counterclock=False,
                wedgeprops=dict(linewidth=1.2, edgecolor="white"),
                textprops=dict(color="black", fontsize=10),
            )
            for t in autotexts:
                t.set_fontsize(18)

            # Donut hole
            centre_circle = plt.Circle((0, 0), 0.50, fc="white")
            ax.add_artist(centre_circle)
            ax.axis("equal")

            # Center text
            ax.text(0, 0.08, method, ha="center", va="center", fontsize=18, weight="bold")
            if env:
                ax.text(0, -0.08, env, ha="center", va="center", fontsize=16)

            if side_idx == 0:
                wedges_for_left = {n: w for n, w in zip(names, wedges)}

        # Build a single combined legend at the figure level.
        # For each group (in ORDER), show both agents' percentages if present; otherwise skip.
        # Use the left plot's wedge colors for handles (create dummy patches when absent).
        # Build a single combined legend at the figure level.
        legend_handles = []
        legend_labels = []
        for g in ORDER:
            if g not in per_agent_pct:
                continue

            # Prefer a real wedge handle from left; else make dummy patch with same color
            if wedges_for_left and g in wedges_for_left:
                handle = wedges_for_left[g]
            else:
                handle = plt.Line2D([0], [0], marker='o', linestyle='', markersize=10,
                                    markerfacecolor=COLORS[g], markeredgecolor='white')
            legend_handles.append(handle)
            legend_labels.append(g)

        # No figure title
        # fig.suptitle(...)  <-- removed

        # Single legend centered below plots (labels only)
        leg = fig.legend(
            legend_handles, legend_labels,
            loc="lower center",
            ncol=2,
            frameon=True,
            fancybox=True,
            borderpad=0.4, handlelength=1.2, handletextpad=0.6, columnspacing=1.0,
            prop={"size": 13},
        )

        # Save
        out = os.path.join(args.output_dir, f"toolcalls_pie_pairs_{env_label.lower()}.svg")
        plt.savefig(out, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"[INFO] Saved paired pie: {out}")

# ────────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2: Average TOTAL tool calls per task for each agent (print)
# ────────────────────────────────────────────────────────────────────────────────

def print_robot_skill_call_stats(calls_df: pd.DataFrame, success_df: pd.DataFrame):
    """
    SUCCESS-ONLY version.

    For each agent and variant (interactive vs noninteractive),
    compute average # of robot-skill tool calls per *successful* run.

    Categories:
      - perception:   robot_detect
      - navigation:   robot_navigate
      - manipulation: robot_pick + robot_open
      - overall:      any 'robot_*'

    A 'run' is (agent, task_type, uid, variant).
    Only runs with success==1 are included in the averaging domain.
    Successful runs that never invoked a category are counted with 0.
    """
    if calls_df.empty:
        print("[WARN] No tool-call data for robot skill stats.")
        return
    if success_df.empty or "success" not in success_df.columns:
        print("[WARN] No success data; cannot compute success-only stats.")
        return

    # --- Domain of runs: successful only ---
    runs_success = (
        success_df.query("success == 1")[["agent", "task_type", "uid", "variant"]]
        .drop_duplicates()
    )
    if runs_success.empty:
        print("[INFO] No successful runs found; nothing to report.")
        return

    # Helper to count tool calls per successful run for a mask
    def _count_mask(mask: pd.Series, colname: str) -> pd.DataFrame:
        return (
            calls_df[mask]
            .merge(runs_success, on=["agent", "task_type", "uid", "variant"], how="inner")
            .groupby(["agent", "task_type", "uid", "variant"], as_index=False)
            .size()
            .rename(columns={"size": colname})
        )

    # Category masks
    m_detect = calls_df["tool_name"].eq("robot_detect")
    m_nav    = calls_df["tool_name"].eq("robot_navigate")
    m_manip  = calls_df["tool_name"].isin(["robot_pick", "robot_open"])
    m_robot  = calls_df["tool_name"].str.startswith("robot_", na=False)

    # Counts per successful run
    df_detect = _count_mask(m_detect, "n_detect")
    df_nav    = _count_mask(m_nav,    "n_navigate")
    df_manip  = _count_mask(m_manip,  "n_manip")
    df_robot  = _count_mask(m_robot,  "n_robot_total")

    # Merge onto the successful-run universe and fill zeros where a successful run didn't call a skill
    merged = runs_success.merge(df_detect, on=["agent", "task_type", "uid", "variant"], how="left") \
                         .merge(df_nav,    on=["agent", "task_type", "uid", "variant"], how="left") \
                         .merge(df_manip,  on=["agent", "task_type", "uid", "variant"], how="left") \
                         .merge(df_robot,  on=["agent", "task_type", "uid", "variant"], how="left")
    for col in ["n_detect", "n_navigate", "n_manip", "n_robot_total"]:
        merged[col] = merged[col].fillna(0).astype(float)

    # --- Overall per agent (success-only averages) ---
    avg_overall = (
        merged.groupby("agent")[["n_detect", "n_navigate", "n_manip", "n_robot_total"]]
        .mean()
        .reset_index()
    )
    n_success_per_agent = merged.groupby("agent").size().to_dict()

    print("\n=== Average robot-skill calls per run (SUCCESS-ONLY, overall) ===")
    print(f"{'agent':24s}  {'perception':>10s}  {'navigation':>10s}  {'manip.':>10s}  {'overall':>10s}  {'n_succ':>7s}")
    for _, r in avg_overall.iterrows():
        n_succ = int(n_success_per_agent.get(r["agent"], 0))
        print(f"{str(r['agent']):24s}  "
              f"{r['n_detect']:>10.2f}  {r['n_navigate']:>10.2f}  {r['n_manip']:>10.2f}  {r['n_robot_total']:>10.2f}  {n_succ:>7d}")

    # --- Per agent × variant (success-only averages) ---
    avg_split = (
        merged.groupby(["agent", "variant"])[["n_detect", "n_navigate", "n_manip", "n_robot_total"]]
        .mean()
        .reset_index()
    )
    n_success_per_agent_variant = merged.groupby(["agent", "variant"]).size().to_dict()

    print("\n=== Average robot-skill calls per run (SUCCESS-ONLY, by agent × variant) ===")
    print(f"{'agent':24s}  {'variant':>13s}  {'perception':>10s}  {'navigation':>10s}  {'manip.':>10s}  {'overall':>10s}  {'n_succ':>7s}")
    for _, r in avg_split.iterrows():
        n_succ = int(n_success_per_agent_variant.get((r["agent"], r["variant"]), 0))
        print(f"{str(r['agent']):24s}  {r['variant']:>13s}  "
              f"{r['n_detect']:>10.2f}  {r['n_navigate']:>10.2f}  {r['n_manip']:>10.2f}  {r['n_robot_total']:>10.2f}  {n_succ:>7d}")

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
    
    success_df = build_success_df(args, tasks)
    # Replace old call with success-only version:
    print_robot_skill_call_stats(calls_df, success_df)

if __name__ == "__main__":
    main()
