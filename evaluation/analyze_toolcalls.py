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
    For each agent: donut chart of tool-call distribution.

    Default: show individual tools (top_k + 'Other').
    If args.family_only: aggregate by family (robot/search/inspect/meta/terminate/other).
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

        if args.family_only:
            # ----- aggregate by family -----
            names = sub["tool_name"].map(_tool_family).value_counts().index.tolist()
            counts = sub["tool_name"].map(_tool_family).value_counts().astype(int)
            values = counts.values
            # colors: one fixed shade per family
            def _fam_color(f):
                if f == "other":
                    return (0.83, 0.83, 0.83, 1.0)
                return _FAMILY_PALETTES.get(f, plt.cm.tab20c)(0.6)
            colors = [_fam_color(f) for f in names]
            # legend_title = "Family (count, %)"
            legend_title = "Tool (count, %)"
            fname_suffix = "_family"
        else:
            # ----- individual tools (top_k + 'Other') -----
            counts = sub["tool_name"].value_counts()
            if len(counts) > top_k:
                top = counts.iloc[:top_k]
                other_sum = counts.iloc[top_k:].sum()
                if other_sum > 0:
                    counts = pd.concat([top, pd.Series({"Other": int(other_sum)})])
                else:
                    counts = top
            counts = counts.astype(int)
            names = counts.index.tolist()
            values = counts.values
            # shade within family for related tools; 'Other' gray
            fam_lists: Dict[str, List[str]] = defaultdict(list)
            for nm in names:
                fam_lists[_tool_family(nm) if nm != "Other" else "other"].append(nm)
            fam_shades = {fam: np.linspace(0.35, 0.85, len(nlist)) for fam, nlist in fam_lists.items()}
            colors, fam_seen_idx = [], defaultdict(int)
            for nm in names:
                if nm == "Other":
                    colors.append((0.83, 0.83, 0.83, 1.0))
                else:
                    fam = _tool_family(nm)
                    palette = _FAMILY_PALETTES.get(fam, plt.cm.tab20c)
                    i = fam_seen_idx[fam]; shade = fam_shades[fam][i]
                    colors.append(palette(shade))
                    fam_seen_idx[fam] += 1
            legend_title = "Tool (count, %)"
            fname_suffix = ""

        total = int(np.sum(values))
        if total == 0:
            print(f"[INFO] No tool calls for agent '{agent}', skipping pie.")
            continue

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
            pctdistance=0.70,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(linewidth=1.2, edgecolor="white"),
            textprops=dict(color="black", fontsize=10)
        )
        # Make the percentage labels larger (and optional: bold)
        for t in autotexts:
            t.set_fontsize(16)       
            # t.set_fontweight("bold")
        # Donut hole
        centre_circle = plt.Circle((0, 0), 0.50, fc="white")
        ax.add_artist(centre_circle)

        # ── counts for center text ─────────────────────────────────────────────
        # Unique tasks (variant-agnostic): base_task_type + uid
        n_tasks = sub.drop_duplicates(["base_task_type", "uid"]).shape[0]
        # If you want per-run (variant-aware) instead, use:
        # n_tasks = sub.drop_duplicates(["task_type", "uid"]).shape[0]

        # Two-line center label
        ax.text(0, 0.06, f"{n_tasks} Tasks", ha="center", va="center", fontsize=16, weight="bold")
        ax.text(0, -0.06, f"(n={total})",    ha="center", va="center", fontsize=13)

        ax.axis("equal")

        # fig.suptitle(
        #     f"Tool-call distribution — { _pretty_agent(agent) }   (n={total})",
        #     x=0.06, ha="left", fontsize=14
        # )

        pct = (pd.Series(values, index=names) / total * 100).round(1)
        legend_labels = [f"{nm} ({int(cnt)}, {pct[nm]}%)" for nm, cnt in zip(names, values)]
        leg = fig.legend(
            wedges, legend_labels,
            title=legend_title,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=2,
            frameon=True,
            fancybox=True, 
            borderpad=0.4, handlelength=1.2, handletextpad=0.6, columnspacing=1.0
        )

        # Style the legend title
        t = leg.get_title()
        t.set_fontsize(16)          # size
        t.set_fontweight("medium")

        fig.subplots_adjust(top=0.86, bottom=0.16, left=0.06, right=0.98)

        out = os.path.join(args.output_dir, f"toolcalls_pie_{agent}{fname_suffix}.png")
        plt.tight_layout()
        plt.savefig(out, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"[INFO] Saved pie: {out}")


# ────────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2: Average TOTAL tool calls per task for each agent (print)
# ────────────────────────────────────────────────────────────────────────────────

def print_robot_skill_call_stats(calls_df: pd.DataFrame):
    """
    For each agent and variant (interactive vs noninteractive),
    compute average # of robot-skill tool calls per run.

    Categories:
      - perception:   robot_detect
      - navigation:   robot_navigate
      - manipulation: robot_pick + robot_open
      - overall:      any 'robot_*'

    A 'run' is (agent, task_type, uid, variant).
    Runs with zero calls in a category are included with count=0.
    """
    if calls_df.empty:
        print("[WARN] No tool-call data for robot skill stats.")
        return

    # Universe of runs so we keep zeros
    runs = calls_df[["agent", "task_type", "uid", "variant"]].drop_duplicates()

    def _count_mask(mask: pd.Series, colname: str) -> pd.DataFrame:
        return (
            calls_df[mask]
            .groupby(["agent", "task_type", "uid", "variant"], as_index=False)
            .size()
            .rename(columns={"size": colname})
        )

    # Category masks
    m_detect = calls_df["tool_name"].eq("robot_detect")
    m_nav    = calls_df["tool_name"].eq("robot_navigate")
    m_manip  = calls_df["tool_name"].isin(["robot_pick", "robot_open"])
    m_robot  = calls_df["tool_name"].str.startswith("robot_", na=False)

    # Counts per run per category
    df_detect = _count_mask(m_detect, "n_detect")
    df_nav    = _count_mask(m_nav,    "n_navigate")
    df_manip  = _count_mask(m_manip,  "n_manip")
    df_robot  = _count_mask(m_robot,  "n_robot_total")

    # Merge all onto the run universe, fill zeros
    merged = runs.merge(df_detect, on=["agent", "task_type", "uid", "variant"], how="left") \
                 .merge(df_nav,    on=["agent", "task_type", "uid", "variant"], how="left") \
                 .merge(df_manip,  on=["agent", "task_type", "uid", "variant"], how="left") \
                 .merge(df_robot,  on=["agent", "task_type", "uid", "variant"], how="left")

    for col in ["n_detect", "n_navigate", "n_manip", "n_robot_total"]:
        merged[col] = merged[col].fillna(0).astype(float)

    # === Overall per agent ===
    avg_overall = (
        merged.groupby("agent")[["n_detect", "n_navigate", "n_manip", "n_robot_total"]]
        .mean()
        .reset_index()
    )
    print("\n=== Average robot-skill calls per run (overall) ===")
    print(f"{'agent':24s}  {'perception':>10s}  {'navigation':>10s}  {'manip.':>10s}  {'overall':>10s}")
    for _, r in avg_overall.iterrows():
        print(f"{str(r['agent']):24s}  "
              f"{r['n_detect']:>10.2f}  {r['n_navigate']:>10.2f}  {r['n_manip']:>10.2f}  {r['n_robot_total']:>10.2f}")

    # === Per agent × variant ===
    avg_split = (
        merged.groupby(["agent", "variant"])[["n_detect", "n_navigate", "n_manip", "n_robot_total"]]
        .mean()
        .reset_index()
    )
    print("\n=== Average robot-skill calls per run (by agent × variant) ===")
    print(f"{'agent':24s}  {'variant':>13s}  {'perception':>10s}  {'navigation':>10s}  {'manip.':>10s}  {'overall':>10s}")
    for _, r in avg_split.iterrows():
        print(f"{str(r['agent']):24s}  {r['variant']:>13s}  "
              f"{r['n_detect']:>10.2f}  {r['n_navigate']:>10.2f}  {r['n_manip']:>10.2f}  {r['n_robot_total']:>10.2f}")

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
    print_robot_skill_call_stats(calls_df)

if __name__ == "__main__":
    main()
