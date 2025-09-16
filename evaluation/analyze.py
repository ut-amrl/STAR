#!/usr/bin/env python3
import argparse
import os
import json
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.container import ErrorbarContainer


# ───────────────────────────────────────────────────────────────
# Fields we want to keep from every result file
# ───────────────────────────────────────────────────────────────
_SUCCESS_FLAGS = [
    "reference_resolution_successs",
    "retrieval_grounding_success",
    "latest_retrieval_success",
    "last_known_state_success",
    "success",
]

_OBJECT_FLAGS = [
    "instance_class",
    "instance_name",
]

# Agent display names (kept if you need to print them verbosely)
_AGENT_DISPLAY = {
    "random":              "Random",
    "sg":                  "SG+S",
    "low_level_gt":        "TR (Oracle Cap.)",
    "replan_low_level_gt": "STAR (Oracle Cap.)",
    "low_level_caption":   "TR (Real Cap.)",
    "replan_low_level_caption": "STAR (Real Cap.)",
}

# Task labels (kept from your original script for pretty xticks)
_TASK_DISPLAY_TWO_LINES = {
    "classonly":        "class-\nbased",
    # "common_sense":     "common\nsense",
    "unambiguous":      "attribute-\nbased",
    "spatial_temporal": "spatial-\ntemporal",
    "spatial":          "spatial",
    "frequency":        "spatial-\nfreqentist",
}
_TASK_DISPLAY = {
    "classonly":        "class-based",
    # "common_sense":     "common-sense",
    "unambiguous":      "attribute-based",
    "spatial_temporal": "spatial-temporal",
    "spatial":          "spatial",
    "frequency":        "spatial-freqentist",
}

# Desired left-to-right task order (raw names!)
_TASK_ORDER = [
    "classonly",
    # "common_sense",
    "unambiguous",
    "spatial",
    "spatial_temporal",
    "frequency",
]

# ───────────────────────────────────────────────────────────────
# NEW: Visual policy (colors/hatches)
# - random: gray
# - sg: purple
# - TR family (low_level_*): blue family
# - STAR family (replan_low_level_*): orange family
# - *_caption: same color family but with dotted hatch
# ───────────────────────────────────────────────────────────────
COLOR_RANDOM = "#9e9e9e"
COLOR_SG = "#8762a2"          # purple
COLOR_TR = "#4a90d9"          # blue family
COLOR_STAR = "#ff9d4d"        # orange family
HATCH_CAPTION = ".."          # dotted hatch for *_caption

def _agent_family(agent: str) -> str:
    """Return 'random', 'sg', 'tr', or 'star'."""
    if agent == "random":
        return "random"
    if agent == "sg":
        return "sg"
    if agent.startswith("replan_low_level"):
        return "star"
    if agent.startswith("low_level"):
        return "tr"
    return "other"

def _is_caption(agent: str) -> bool:
    return agent.endswith("_caption")

def _style_for_agent(agent: str) -> dict:
    fam = _agent_family(agent)
    color = {
        "random": COLOR_RANDOM,
        "sg": COLOR_SG,
        "tr": COLOR_TR,
        "star": COLOR_STAR,
    }.get(fam, "#999999")
    hatch = HATCH_CAPTION if _is_caption(agent) else ""
    return {"color": color, "hatch": hatch}

# ───────────────────────────────────────────────────────────────
# Utilities (as in your original script)
# ───────────────────────────────────────────────────────────────
def _task_sort_key(t):
    try:
        return _TASK_ORDER.index(t)
    except ValueError:
        return len(_TASK_ORDER) + hash(t)

def _pretty_task(raw: str, oneline: bool = False) -> str:
    """Human-friendly label for legends / tick-labels."""
    task_display = _TASK_DISPLAY if oneline else _TASK_DISPLAY_TWO_LINES
    for orig, nice in task_display.items():
        if raw.startswith(orig):
            return f"{nice}{raw[len(orig):]}"
    return raw

def _pretty_agent(raw: str) -> str:
    for orig, nice in _AGENT_DISPLAY.items():
        if raw.startswith(orig):
            return f"{nice}{raw[len(orig):]}"
    return raw

def _strip(record: dict) -> dict:
    """Return only the fields listed in _SUCCESS_FLAGS and _OBJECT_FLAGS (missing keys → None)."""
    return {k: record.get(k) for k in _SUCCESS_FLAGS + _OBJECT_FLAGS}

# Wilson 95% CI half-width helper
def _wilson_halfwidth(succ: float, n: int, z: float = 1.96) -> float:
    """95% Wilson interval half-width for a binomial proportion."""
    if n <= 0:
        return np.nan
    p = succ / n
    denom = 1.0 + (z**2) / n
    return z * np.sqrt(p * (1 - p) / n + (z**2) / (4 * n * n)) / denom

def _nice_ylim(max_height_with_err: float, pad: float = 0.02, clamp: bool = False):
    top = max_height_with_err + pad
    if clamp:
        top = min(1.0, top)
    return (0.0, top)

def _mix(c1, c2, a):  # linear blend in RGB
    r1, r2 = np.array(mcolors.to_rgb(c1)), np.array(mcolors.to_rgb(c2))
    return mcolors.to_hex((1 - a) * r1 + a * r2)

def _darken(hex_color, amount=0.25):
    """Darker shade of a color (amount in [0,1])."""
    return _mix(hex_color, "#000000", amount)

def _err_style(ecolor="#333333"):
    """Unified, nicer errorbar style."""
    return dict(fmt="none", ecolor=ecolor, elinewidth=1.2, capsize=4, capthick=1.2, alpha=0.9, zorder=4)

def _round_errcaps(eb: ErrorbarContainer):
    """Round line caps & joins for prettier whiskers."""
    try:
        # Vertical whiskers (LineCollection)
        for lc in eb[2]:
            lc.set_capstyle("round")
            lc.set_joinstyle("round")
        # Horizontal caps (Line2D objects)
        for cap in eb[1]:
            cap.set_solid_capstyle("round")
            cap.set_solid_joinstyle("round")
    except Exception:
        pass
    
def _append_overall_row_for_all_tasks(df, agg, agent_order, yerr_map=None):
    """
    Appends an 'overall' row to `agg` (index) with mean success per agent across ALL tasks in df.
    If yerr_map is provided, adds Wilson 95% CI half-widths for ('overall', agent).
    """
    import pandas as pd

    # mean success per agent across all tasks present in df
    overall_series = (
        df[["agent", "success"]]
        .dropna(subset=["success"])
        .groupby("agent")["success"]
        .mean()
        .reindex(agent_order)
    )
    agg.loc["overall"] = overall_series

    if yerr_map is not None:
        stats_overall = (
            df[["agent", "success"]]
            .dropna(subset=["success"])
            .groupby("agent")["success"]
            .agg(n="count", s="sum")
            .reindex(agent_order)
        )
        for ag, row in stats_overall.dropna().iterrows():
            yerr_map[("overall", ag)] = _wilson_halfwidth(int(row["s"]), int(row["n"]))
    return agg, yerr_map


def _append_overall_row_for_interactive(dfi, agg, agent_order, yerr_map=None):
    """
    Appends an 'overall' row to `agg` using only *interactive* rows (dfi).
    If yerr_map is provided, adds Wilson 95% CI half-widths for ('overall', agent).
    """
    import pandas as pd

    overall_series = (
        dfi[["agent", "success"]]
        .dropna(subset=["success"])
        .groupby("agent")["success"]
        .mean()
        .reindex(agent_order)
    )
    agg.loc["overall"] = overall_series

    if yerr_map is not None:
        stats_overall = (
            dfi[["agent", "success"]]
            .dropna(subset=["success"])
            .groupby("agent")["success"]
            .agg(n="count", s="sum")
            .reindex(agent_order)
        )
        for ag, row in stats_overall.dropna().iterrows():
            yerr_map[("overall", ag)] = _wilson_halfwidth(int(row["s"]), int(row["n"]))
    return agg, yerr_map

    
def print_common_sense_results(args, df):
    """
    Print success rates for 'common_sense' tasks.
    If 'common_sense_interactive' exists, print that as well.
    Respects --error_bars using Wilson 95% CI (same helper as plots).
    """
    import pandas as pd

    if df.empty or "task_type" not in df.columns or "success" not in df.columns:
        print("[WARN] No data available to report common_sense results.")
        return

    variants = [("common_sense", "Common-sense"),
                ("common_sense_interactive", "Common-sense (interactive)")]

    for raw_task, pretty_name in variants:
        sub = df[df["task_type"] == raw_task].copy()
        if sub.empty:
            # silent skip if that variant doesn't exist in this run
            continue

        # Aggregate by agent
        agg = (
            sub.groupby("agent")["success"]
               .agg(mean="mean", n="count", s="sum")
               .reindex(args.agent_types)
        )

        print(f"\n=== {pretty_name}: Success Rate by Agent ===")
        for ag, row in agg.dropna(subset=["mean"]).iterrows():
            val = float(row["mean"])
            out = f"{_pretty_agent(ag)}: {val:.2f}"
            if args.error_bars:
                yerr = _wilson_halfwidth(int(row["s"]), int(row["n"]))
                if np.isfinite(yerr):
                    out += f" ± {yerr:.2f}"
            print(out)
        print("===========================================\n")

# ───────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="evaluation/sim_outputs/")
    parser.add_argument("--output_dir", type=str, default="evaluation/sim_outputs/")
    parser.add_argument("--task_config", type=str, default="evaluation/config/tasks_sim_all.txt")
    parser.add_argument(
        "--agent_types", nargs="+",
        default=["random", "sg", "low_level_gt", "replan_low_level_gt", "low_level_caption", "replan_low_level_caption"]
    )
    parser.add_argument("--error_bars", action="store_true",
                        help="If set, draw 95% Wilson CI error bars for success rates.")
    parser.add_argument("--cascade_error_analysis", action="store_true",
                        help="If set, perform cascade failure analysis.")
    parser.add_argument("--with_overall", action="store_true",
                        help="Include an 'overall' column aggregating across tasks.")
    return parser.parse_args()

# ───────────────────────────────────────────────────────────────
# IO
# ───────────────────────────────────────────────────────────────
def _load_json_safe(path: str):
    """Return parsed dict if file exists, else None."""
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error in {path}: {e}")
    return None

def parse_task_config(task_config: str) -> Dict[str, List[str]]:
    """
    Parse tasks_sim.txt into {task_type: [hash1, hash2, ...]}.
    Expected line format (1 per task, blanks/comments allowed):
        spatial_temporal/09b7d427.json
        frequency/032e0e9a.json
        # comment lines are ignored
    """
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

            hash_name, ext = os.path.splitext(os.path.basename(fname))
            if ext != ".json":
                print(f"[WARN] Expected .json, got {ext} in: {line}")
            tasks[task_type].append(hash_name)
    return dict(tasks)

def parse_results(args, tasks):
    results = defaultdict(dict)
    for task_type, hashes in tasks.items():
        for h in hashes:
            entry = {}
            for agent in args.agent_types:
                fname = f"results_{agent}_{h}.json"
                path = os.path.join(args.data_dir, task_type, h, fname)

                raw = _load_json_safe(path)
                entry[agent] = _strip(raw) if raw is not None else None

            results[task_type][h] = entry
    return results

def analyze_results(
    args,
    results: Dict[str, Dict[str, Dict[str, Any]]],
    agent_types: List[str],
):
    import pandas as pd

    rows = []
    for task_type, task_dict in results.items():
        for h, per_agent in task_dict.items():
            for agent in agent_types:
                rec = per_agent.get(agent)
                if rec is None:
                    continue
                row = {"task_type": task_type, "hash": h, "agent": agent}
                row.update(rec)
                rows.append(row)

    df = pd.DataFrame(rows)

    rates: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for agent in agent_types:
        sub = df[df["agent"] == agent]
        for task_type, g in sub.groupby("task_type"):
            for flag in _SUCCESS_FLAGS:
                if flag not in g.columns:
                    continue
                denom = g[flag].notna().sum()
                if denom:
                    rates[agent][task_type][flag] = g[flag].mean()

    # pretty-print
    # for ag, per_task in rates.items():
    #     print(f"\n========== {ag} ==========")
    #     for task_t, stats in per_task.items():
    #         print(f"\n--- {task_t} ---")
    #         for flag, r in stats.items():
    #             print(f"{flag:32s}: {r:.1%}")

    return rates, df

# ───────────────────────────────────────────────────────────────
# Plot helpers / legends
# ───────────────────────────────────────────────────────────────
def _right_side_combined_legend(ax):
    """
    Two-block legend to the RIGHT of the axes:
      Block 1 (Approach): Random, SG+S, TR+S, STAR  [color-coded]
      Block 2 (Knowledge Regime): Oracle, Realistic       [hatch-coded]
    """
    # ---- Block 1: Approach (colors) ----
    approach_handles = [
        mpatches.Patch(facecolor=COLOR_RANDOM, edgecolor="black", linewidth=0.7, label="Random"),
        mpatches.Patch(facecolor=COLOR_SG,     edgecolor="black", linewidth=0.7, label="SG+S"),
        mpatches.Patch(facecolor=COLOR_TR,     edgecolor="black", linewidth=0.7, label="TR+S"),
        mpatches.Patch(facecolor=COLOR_STAR,   edgecolor="black", linewidth=0.7, label="STAR"),
    ]
    leg_approach = ax.legend(
        handles=approach_handles,
        title="Approach",
        title_fontsize=12,
        fontsize=11,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),   # right of plot, near top
        frameon=True, framealpha=0.95, facecolor="white", edgecolor="lightgray",
        borderaxespad=0.0
    )
    leg_approach.get_title().set_fontweight("medium")
    ax.add_artist(leg_approach)

    # ---- Block 2: Knowledge Regime (hatches) ----
    regime_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.7, label="Oracle",   hatch=""),
        mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.7, label="Realistic", hatch=HATCH_CAPTION),
    ]
    leg_regime = ax.legend(
        handles=regime_handles,
        title="Env. Knowledge",
        title_fontsize=12,
        fontsize=11,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.65),   # right of plot, a bit lower
        frameon=True, framealpha=0.95, facecolor="white", edgecolor="lightgray",
        borderaxespad=0.0
    )
    leg_regime.get_title().set_fontweight("medium")
    return leg_approach, leg_regime

# ───────────────────────────────────────────────────────────────
# Plots
# ───────────────────────────────────────────────────────────────
def plot_overall_success(args, df):
    _TASK_DISPLAY_TWO_LINES["overall"] = "overall"
    _TASK_DISPLAY["overall"] = "overall"
    
    """
    Colors encode agent family; caption agents are dot-hatched.
    Legend is placed to the RIGHT (combined: color roles + caption type).
    """
    import pandas as pd
    import matplotlib as mpl
    mpl.rcParams["hatch.linewidth"] = 0.8

    if df.empty or "success" not in df.columns:
        print("[WARN] No data for overall-success plot")
        return

    # Error bar stats (Wilson)
    yerr_map = {}
    if args.error_bars:
        stats_gb = (
            df[["task_type", "agent", "success"]]
              .dropna(subset=["success"])
              .groupby(["task_type", "agent"])
        )
        _stats = stats_gb.agg(mean=("success", "mean"),
                              n=("success", "count"),
                              s=("success", "sum")).reset_index()
        yerr_map = {
            (r["task_type"], r["agent"]): _wilson_halfwidth(r["s"], int(r["n"]))
            for _, r in _stats.iterrows()
        }

    # Means
    agg = (
        df.groupby(["task_type", "agent"])["success"]
          .mean()
          .reset_index()
          .pivot(index="task_type", columns="agent", values="success")
          .reindex(columns=args.agent_types)
          .sort_index()
    )
    if agg.empty:
        print("[WARN] Aggregation produced empty frame")
        return

    tasks = [t for t in _TASK_ORDER if t in agg.index]
    if args.with_overall:
        agg, yerr_map = _append_overall_row_for_all_tasks(df, agg, args.agent_types,
                                                          yerr_map if args.error_bars else None)
        tasks = tasks + (["overall"])
    if not tasks:
        print("[WARN] No task types from _TASK_ORDER found in data — skipping")
        return

    n_tasks  = len(tasks)
    n_agents = len(args.agent_types)
    x = np.arange(n_tasks)
    width = 0.8 / max(1, n_agents)

    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    max_with_err = 0.0

    for i_agent, ag in enumerate(args.agent_types):
        st = _style_for_agent(ag)
        color, hatch = st["color"], st["hatch"]
        for i_task, task in enumerate(tasks):
            val = agg.loc[task, ag] if ag in agg.columns else np.nan
            if pd.isna(val):
                continue
            val = float(val)
            xpos = x[i_task] + (i_agent - (n_agents - 1) / 2.0) * width

            bar = ax.bar(
                xpos, val, width=width,
                color=color, edgecolor="black", linewidth=0.7, zorder=2.5,
            )[0]
            if hatch:
                bar.set_hatch(hatch)
            bar.set_linewidth(0.7)

            # error bars match color (slightly darker)
            yerr = None
            if args.error_bars:
                yerr = yerr_map.get((task, ag))
                if yerr is not None and np.isfinite(yerr):
                    ec = _darken(color, 0.30)
                    eb = ax.errorbar(xpos, val, yerr=yerr, **_err_style(ec))
                    _round_errcaps(eb)

            top = val + (float(yerr) if (yerr is not None and np.isfinite(yerr)) else 0.0)
            max_with_err = max(max_with_err, top)

    # Axes, ticks, ylim
    ax.set_ylabel("Execution Success Rate", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_task(t) for t in tasks], rotation=0, ha="center")
    # ax.set_xticklabels([_pretty_task(t) for t in tasks], rotation=45, ha="right")
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.set_ylim(*_nice_ylim(max_with_err, pad=0.05, clamp=False))

    # Shade alternate task bins
    ax.set_xlim(-0.5, n_tasks - 0.5)
    for i in range(0, n_tasks, 2):
        ax.axvspan(i - 0.5, i + 0.5, color="#b7b7b7", alpha=0.12, zorder=0)

    # Combined legend to the RIGHT
    _right_side_combined_legend(ax)

    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42      # TrueType in PDFs (Type 42)
    mpl.rcParams['ps.fonttype']  = 42
    mpl.rcParams['text.usetex']  = False   # usetex often yields Type 3 via dvipng
    mpl.rcParams['svg.fonttype'] = 'none'  # keep SVG text as text (not paths)
    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    out_png = os.path.join(args.output_dir, "results_main.png")
    out_svg = os.path.join(args.output_dir, "results_main.svg")
    out_pdf = os.path.join(args.output_dir, "results_main.pdf")
    plt.savefig(out_pdf, bbox_inches="tight") 
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close()
    print(f"[INFO] interactive-success plot saved to {out_png} and {out_svg} and {out_pdf}")
    
    # ── NEW: print table of results ──────────────────────────────
    import pandas as pd
    print("\n=== Overall Success Rates (per task × agent) ===")
    if args.error_bars:
        # Reuse yerr_map if available
        for task in tasks:
            row_strs = []
            for ag in args.agent_types:
                val = agg.loc[task, ag] if ag in agg.columns else np.nan
                if pd.isna(val):
                    continue
                yerr = yerr_map.get((task, ag))
                row_strs.append(f"{_pretty_agent(ag)}: {val:.2f} ± {yerr:.2f}" if yerr else f"{_pretty_agent(ag)}: {val:.2f}")
            print(f"{_pretty_task(task, oneline=True):20s} | " + " | ".join(row_strs))
    else:
        for task in tasks:
            row_strs = []
            for ag in args.agent_types:
                val = agg.loc[task, ag] if ag in agg.columns else np.nan
                if pd.isna(val):
                    continue
                row_strs.append(f"{_pretty_agent(ag)}: {val:.2f}")
            print(f"{_pretty_task(task, oneline=True):20s} | " + " | ".join(row_strs))
    print("================================================\n")

def _base_task(task: str, suffix: str = "_interactive") -> str:
    """Map 'classonly_interactive' -> 'classonly'; pass through otherwise."""
    return task[:-len(suffix)] if task.endswith(suffix) else task

def plot_interactive_success(args, df, suffix: str = "_interactive"):
    """
    Plot success rates for interactive-search tasks only, styled identically to plot_overall_success.
    - Filters rows whose task_type endswith `suffix` (default: '_interactive')
    - Collapses labels to base task names (e.g., 'classonly')
    - Colors encode agent family; caption agents are dot-hatched
    - Combined legend on the right (Approach colors + Env. Knowledge hatch)
    Outputs:
      <output_dir>/results_interactive.png
      <output_dir>/results_interactive.svg
    """
    import pandas as pd
    import matplotlib as mpl
    mpl.rcParams["hatch.linewidth"] = 0.8

    # Basic guards
    if df.empty or "task_type" not in df.columns or "success" not in df.columns:
        print("[WARN] No data for interactive-success plot")
        return

    # Keep only interactive tasks and map to base task
    dfi = df[df["task_type"].astype(str).str.endswith(suffix)].copy()
    if dfi.empty:
        print(f"[INFO] No '*{suffix}' tasks found — skipping interactive plot")
        return
    dfi["base_task"] = dfi["task_type"].map(lambda t: _base_task(t, suffix))

    # Aggregate means per (base_task, agent)
    agg = (
        dfi.groupby(["base_task", "agent"])["success"]
           .mean()
           .reset_index()
           .pivot(index="base_task", columns="agent", values="success")
           .reindex(columns=args.agent_types)
    )
    if agg.empty:
        print("[WARN] Aggregation produced empty frame for interactive plot")
        return

    # Order tasks to match global order when present
    tasks_present = [t for t in _TASK_ORDER if t in agg.index]
    if not tasks_present:
        tasks_present = list(agg.index)
    agg = agg.loc[tasks_present]

    # Error bars (Wilson 95%)
    yerr_map = {}
    if args.error_bars:
        stats_gb = (
            dfi[["base_task", "agent", "success"]]
               .dropna(subset=["success"])
               .groupby(["base_task", "agent"])
        )
        _stats = stats_gb.agg(
            mean=("success", "mean"),
            n=("success", "count"),
            s=("success", "sum")
        ).reset_index()
        yerr_map = {
            (r["base_task"], r["agent"]): _wilson_halfwidth(r["s"], int(r["n"]))
            for _, r in _stats.iterrows()
        }

    # Draw bars (same geometry & cosmetics as overall)
    n_tasks  = len(tasks_present)
    n_agents = len(args.agent_types)
    x = np.arange(n_tasks)
    width = 0.8 / max(1, n_agents)

    fig, ax = plt.subplots(figsize=(12.5, 6))
    max_with_err = 0.0

    for i_agent, ag in enumerate(args.agent_types):
        st = _style_for_agent(ag)
        color, hatch = st["color"], st["hatch"]

        for i_task, task in enumerate(tasks_present):
            val = agg.loc[task, ag] if ag in agg.columns else np.nan
            if pd.isna(val):
                continue
            val = float(val)
            xpos = x[i_task] + (i_agent - (n_agents - 1) / 2.0) * width

            bar = ax.bar(
                xpos, val, width=width,
                color=color, edgecolor="black", linewidth=0.7, zorder=2.5
            )[0]
            if hatch:
                bar.set_hatch(hatch)
            bar.set_linewidth(0.7)

            # Wilson yerr, styled identically to overall
            yerr = None
            if args.error_bars:
                yerr = yerr_map.get((task, ag))
                if yerr is not None and np.isfinite(yerr):
                    ec = _darken(color, 0.30)
                    eb = ax.errorbar(xpos, val, yerr=yerr, **_err_style(ec))
                    _round_errcaps(eb)

            top = val + (float(yerr) if (yerr is not None and np.isfinite(yerr)) else 0.0)
            max_with_err = max(max_with_err, top)

    # Axes, ticks, grid, ylim — match overall style
    ax.set_ylabel("Execution Success Rate", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_task(t) for t in tasks_present], rotation=45, ha="right")
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.set_ylim(*_nice_ylim(max_with_err, pad=0.05, clamp=False))

    # Shade alternate task bins (same look)
    ax.set_xlim(-0.5, n_tasks - 0.5)
    for i in range(0, n_tasks, 2):
        ax.axvspan(i - 0.5, i + 0.5, color="#b7b7b7", alpha=0.12, zorder=0)

    # Identical combined legend to the RIGHT
    _right_side_combined_legend(ax)

    # Save both PNG and SVG (like overall)
    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    out_png = os.path.join(args.output_dir, "results_interactive.png")
    out_svg = os.path.join(args.output_dir, "results_interactive.svg")
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close()
    print(f"[INFO] interactive-success plot saved to {out_png} and {out_svg}")
    
        # ── NEW: print table of interactive results ─────────────────
    import pandas as pd
    print("\n=== Interactive Success Rates (per base_task × agent) ===")
    if args.error_bars:
        for task in tasks_present:
            row_strs = []
            for ag in args.agent_types:
                val = agg.loc[task, ag] if ag in agg.columns else np.nan
                if pd.isna(val):
                    continue
                yerr = yerr_map.get((task, ag))
                if yerr is not None and np.isfinite(yerr):
                    row_strs.append(f"{_pretty_agent(ag)}: {val:.2f} ± {yerr:.2f}")
                else:
                    row_strs.append(f"{_pretty_agent(ag)}: {val:.2f}")
            print(f"{_pretty_task(task, oneline=True):20s} | " + " | ".join(row_strs))
    else:
        for task in tasks_present:
            row_strs = []
            for ag in args.agent_types:
                val = agg.loc[task, ag] if ag in agg.columns else np.nan
                if pd.isna(val):
                    continue
                row_strs.append(f"{_pretty_agent(ag)}: {val:.2f}")
            print(f"{_pretty_task(task, oneline=True):20s} | " + " | ".join(row_strs))
    print("================================================\n")



def cascade_failure_analysis(df, success_flags=_SUCCESS_FLAGS, group_keys=["task_type", "hash", "agent"]):
    """
    For each cascading success flag, print all task instances that succeeded
    at the previous flag but failed at the current.
    """
    print("\n=== Cascade Failure Analysis ===")
    # Ensure the DataFrame has all necessary columns
    missing = [flag for flag in success_flags if flag not in df.columns]
    if missing:
        print(f"[WARN] Missing columns in dataframe: {missing}")
        return

    # Sort for easier reading
    df_sorted = df.sort_values(group_keys)

    first_flag = success_flags[0]
    print(f"\n--- Tasks that FAILED at '{first_flag}' (first stage) ---")
    mask = (df_sorted[first_flag] == 0)
    failures = df_sorted[mask]
    if failures.empty:
        print("(none)")
    else:
        print(failures[group_keys + [first_flag]])

    for i in range(1, len(success_flags)):
        prev_flag = success_flags[i-1]
        curr_flag = success_flags[i]
        print(f"\n--- Tasks that SUCCEEDED at '{prev_flag}' but FAILED at '{curr_flag}' ---")
        mask = (df_sorted[prev_flag] == 1) & (df_sorted[curr_flag] == 0)
        failures = df_sorted[mask]
        if failures.empty:
            print("(none)")
        else:
            print(failures[group_keys + [prev_flag, curr_flag]])

    print("\n=== End Cascade Failure Analysis ===")

# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    tasks = parse_task_config(args.task_config)
    results = parse_results(args, tasks)

    _, df = analyze_results(args, results, args.agent_types)
    if args.cascade_error_analysis:
        cascade_failure_analysis(df)
    plot_overall_success(args, df)
    plot_interactive_success(args, df)
    print_common_sense_results(args, df)
    # Cascade failure analysis intentionally removed per request

if __name__ == "__main__":
    main()
