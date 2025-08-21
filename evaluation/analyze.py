import argparse
import os
import json
from collections import defaultdict
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# fields we want to keep from every result file
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

_AGENT_DISPLAY = {
    "random": "Random",
    "replan_low_level_gt": "Interleaving Exec.",
    "low_level_gt": "One-shot Exec.",
    "high_level_gt": "high-level",
    # Add more mappings as needed
}

_TASK_DISPLAY_TWO_LINES = {
    "classonly":        "class-\nbased",
    "unambiguous":      "attribute-\nbased",
    "spatial_temporal": "spatial-\ntemporal",
    "spatial":          "spatial",
    "frequency":        "spatial-\nfreqentist",
    # add more renames here if needed
}

_TASK_DISPLAY = {
    "classonly":        "class-based",
    "unambiguous":      "attribute-based",
    "spatial_temporal": "spatial-temporal",
    "spatial":          "spatial",
    "frequency":        "spatial-freqentist",
    # add more renames here if needed
}

_COLOR_GROUP = {
    "classonly":        plt.cm.Oranges,
    "unambiguous":      plt.cm.Greens,
    "spatial_temporal": plt.cm.Blues,
    "spatial":          plt.cm.Purples,
    "frequency":        plt.cm.Greys,
}

# desired left-to-right order in every grouped plot  (raw names!)
_TASK_ORDER = [
    "classonly",       # → “class-only”
    "unambiguous",     # → “attribute-based”
    "spatial",         # → “spatial”
    "spatial_temporal",# → “temporal”
    "frequency",       # → “freqencist”
]

# Agent-specific visual style (kept consistent across all plots)
_AGENT_STYLE = {
    "low_level_gt":       {"color": "#4a90d9", "hatch": "//"},   # lighter blue
    "replan_low_level_gt":{"color": "#ff9d4d", "hatch": ""},     # lighter orange
    "high_level_gt":      {"color": "#58c96e", "hatch": "xx"},   # lighter green
    "random":             {"color": "#9e9e9e", "hatch": ".."},   # lighter gray
}

def _style_for_agent(agent: str) -> dict:
    """Return {'color': hex, 'hatch': str} for an agent; fallback if unknown."""
    return _AGENT_STYLE.get(agent, {"color": "#999999", "hatch": ""})

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
            return f"{nice}{raw[len(orig):]}"     # keep any suffix
    return raw

def _pretty_agent(raw: str) -> str:
    """Human-friendly label for legends / tick-labels."""
    for orig, nice in _AGENT_DISPLAY.items():
        if raw.startswith(orig):
            return f"{nice}{raw[len(orig):]}"     # keep any suffix
    return raw

def _strip(record: dict) -> dict:
    """Return only the fields listed in _SUCCESS_FLAGS and _OBJECT_FLAGS (missing keys → None)."""
    return {k: record.get(k) for k in _SUCCESS_FLAGS + _OBJECT_FLAGS}

# ── NEW: Wilson 95% CI half-width helper ───────────────────────────────────────
def _wilson_halfwidth(succ: float, n: int, z: float = 1.96) -> float:
    """95% Wilson interval half-width for a binomial proportion."""
    if n <= 0:
        return np.nan
    p = succ / n
    denom = 1.0 + (z**2) / n
    return z * np.sqrt(p * (1 - p) / n + (z**2) / (4 * n * n)) / denom

def _err_style():
    """Unified, prettier errorbar style."""
    return dict(
        fmt="none",
        ecolor="#333333",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        alpha=0.9,
        zorder=4,
    )

def _nice_ylim(max_height_with_err: float, pad: float = 0.02):
    """Clamp to [0,1], but give a bit of headroom."""
    top = min(1.0, max(0.0, max_height_with_err) + pad)
    return (0.0, top if top > 0 else 1.0)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="evaluation/sim_outputs/")
    parser.add_argument("--output_dir", type=str, default="evaluation/sim_outputs/")
    parser.add_argument("--task_config", type=str, default="evaluation/config/tasks_sim_all.txt")
    parser.add_argument("--agent_types", nargs="+", default=["random", "low_level_gt", "replan_low_level_gt"])
    # NEW: toggle error bars
    parser.add_argument("--error_bars", action="store_true",
                        help="If set, draw 95% Wilson CI error bars for success rates.")
    parser.add_argument("--cascade_error_analysis", action="store_true", help="If set, perform cascade failure analysis.")
    return parser.parse_args()

def _load_json_safe(path: str):
    """Return parsed dict if file exists, else None."""
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error in {path}: {e}")
    return None

def parse_task_config(task_config: str) -> dict[str, list[str]]:
    """
    Parse tasks_sim.txt into {task_type: [hash1, hash2, ...]}.
    Expected line format (1 per task, blanks/comments allowed):
        spatial_temporal/09b7d427.json
        frequency/032e0e9a.json
        # comment lines are ignored
    """
    tasks: dict[str, list[str]] = defaultdict(list)

    with open(task_config, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue                       # skip blanks & comments
            try:
                task_type, fname = line.split("/", 1)
            except ValueError:                # malformed line → warn & skip
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
    for ag, per_task in rates.items():
        print(f"\n========== {ag} ==========")
        for task_t, stats in per_task.items():
            print(f"\n--- {task_t} ---")
            for flag, r in stats.items():
                print(f"{flag:32s}: {r:.1%}")
                
    return rates, df

def plot_overall_success(args, df):
    import matplotlib as mpl
    import pandas as pd
    mpl.rcParams["hatch.linewidth"] = 0.5

    # Stats for error bars (per task_type, agent)
    yerr_map = {}
    if args.error_bars and not df.empty and "success" in df.columns:
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

    agg = (
        df.groupby(["task_type", "agent"])["success"]
          .mean()
          .reset_index()
          .pivot(index="task_type", columns="agent", values="success")
          .reindex(columns=args.agent_types)           # keep agent order
          .sort_index()
    )

    if agg.empty:
        print("[WARN] No data for overall-success plot")
    else:
        tasks = [t for t in _TASK_ORDER if t in agg.index]
        if not tasks:
            print("[WARN] No task types from _TASK_ORDER found in data — skipping overall plot")
            return
        n_tasks = len(tasks)
        n_agents = len(args.agent_types)
        x = np.arange(n_tasks)
        width = 0.8 / n_agents

        fig, ax = plt.subplots(figsize=(11, 6))

        shade_points = np.linspace(0.25, 0.85, n_agents)    # same as helper

        for i_agent, ag in enumerate(args.agent_types):
            style = _style_for_agent(ag)
            for i_task, task in enumerate(tasks):
                colour = style["color"]
                xpos = x[i_task] + (i_agent - (n_agents-1)/2) * width
                val = agg.loc[task, ag]
                bar = ax.bar(
                    xpos, val, width=width, color=colour, edgecolor="black", linewidth=0.7
                )[0]
                bar.set_hatch(_style_for_agent(ag)["hatch"])
                bar.set_linewidth(0.5)

                # error bar per bar (if enabled)
                if args.error_bars:
                    yerr = yerr_map.get((task, ag))
                    if yerr is not None and np.isfinite(yerr):
                        ax.errorbar(
                            xpos, val, yerr=yerr,
                            fmt="none", ecolor="black",
                            elinewidth=0.8, capsize=3, capthick=0.8,
                            zorder=3
                        )

        # x-tick labels
        ax.set_xticks(x)
        ax.set_xticklabels([_pretty_task(t) for t in tasks], rotation=45, ha="right")

        # y-axis & grid
        ax.set_ylabel("Execution Success Rate", fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.4)
        
        # FULL-BIN gray shading (midpoint to midpoint)
        ax.set_xlim(-0.5, n_tasks - 0.5)
        for i in range(0, n_tasks, 2):  # Shade every other task bin
            left = i - 0.5
            right = i + 0.5
            ax.axvspan(left, right, color="#b7b7b7", alpha=0.15, zorder=0)

        # legend – one patch per agent (colour sampled from first task)
        legend_handles = []
        for ag in args.agent_types:
            st = _style_for_agent(ag)
            legend_handles.append(
                mpatches.Patch(
                    facecolor=st["color"],
                    edgecolor="black",
                    label=_pretty_agent(ag),
                    linewidth=0.7,
                    hatch=st["hatch"],
                )
            )
        ax.legend(handles=legend_handles, title="Agent",
                  bbox_to_anchor=(1.04, 1), loc="upper left")

        ax.set_title("Overall Execution Success by Task Type")

        plt.tight_layout()
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "success.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] overall-success plot saved to {out_path}")

def plot_object_class_success(args, df):
    """
    One figure, x-axis = instance_class, grouped bars = (task_type, agent)
    Colours:  colour-map per task_type, shade per agent.
    """
    if df.empty or "instance_class" not in df.columns:
        print("[WARN] No instance_class info available – skipping object-class plot")
        return

    import numpy as np
    import pandas as pd
    os.makedirs(args.output_dir, exist_ok=True)

    # --- only plot selected agents (drop 'random') -----------------------  # CHANGED
    agents = [ag for ag in args.agent_types if ag != "random"]              # CHANGED
    if not agents:
        print("[WARN] No agents left after excluding 'random' — skipping")
        return

    df_plot = df[df["agent"].isin(agents)].copy()                           # CHANGED

    # ── 1 aggregate success per (instance_class, task_type, agent) ─────────
    plot_df = (
        df_plot.groupby(["instance_class", "task_type", "agent"])["success"] # CHANGED
              .mean()
              .reset_index()
              .pivot(index="instance_class",
                     columns=["task_type", "agent"],
                     values="success")
    )
    if plot_df.empty:
        print("[WARN] Nothing to plot – empty after pivot")
        return

    # ── 1b stats for error bars (from raw df, not the means) ───────────────
    yerr_map = {}
    if args.error_bars:
        stats_gb = (
            df_plot[["instance_class", "task_type", "agent", "success"]]     # CHANGED
                  .dropna(subset=["success"])
                  .groupby(["instance_class", "task_type", "agent"])
        )
        _stats = stats_gb.agg(mean=("success", "mean"),
                              n=("success", "count"),
                              s=("success", "sum")).reset_index()
        yerr_map = {
            (r["instance_class"], r["task_type"], r["agent"]): _wilson_halfwidth(r["s"], int(r["n"]))
            for _, r in _stats.iterrows()
        }

    # ── 2 build display names & colours in one pass ───────────────────────
    flat_cols, colours = [], []
    unique_tasks = {t for t, _ in plot_df.columns}
    ordered_tasks = [t for t in _TASK_ORDER if t in unique_tasks]
    if not ordered_tasks:
        print("[WARN] No task types from _TASK_ORDER found for object-class plot — skipping")
        return

    for task in ordered_tasks:
        for ag in agents:                                                   # CHANGED
            flat_cols.append((task, ag))
            colours.append(_style_for_agent(ag)["color"])

    hatches = [_style_for_agent(agent)["hatch"] for (task, agent) in flat_cols]

    # re-index & flatten columns as before
    plot_df = plot_df.reindex(columns=pd.MultiIndex.from_tuples(flat_cols))
    # Keep a copy of the (task, agent) mapping per column index for error bars
    col_keys = list(plot_df.columns)
    plot_df.columns = [f"{_pretty_task(t, oneline=True)}\n{_pretty_agent(ag)}" for t,ag in col_keys]

    # draw the bars
    ax = plot_df.plot(
        kind="bar",
        rot=45,
        figsize=(11,6),
        color=colours,
        edgecolor="black",
        linewidth=0.5,
        width=0.85,
    )
    
    # how many object-classes (i.e. rows in plot_df)?
    n_rows = plot_df.shape[0]

    # now, for each column j, the next n_rows patches are that column's bars:
    for col_idx, ((task, agent), hatch) in enumerate(zip(col_keys, hatches)):
        start = col_idx * n_rows
        end   = start + n_rows
        for row_offset, patch in enumerate(ax.patches[start:end]):
            patch.set_hatch(hatch)
            # error bar per bar (if enabled)
            if args.error_bars:
                instance_class = plot_df.index[row_offset]   # row label
                yerr = yerr_map.get((instance_class, task, agent))
                if yerr is not None and np.isfinite(yerr):
                    cx = patch.get_x() + patch.get_width()/2.0
                    cy = patch.get_height()
                    ax.errorbar(
                        cx, cy, yerr=yerr,
                        fmt="none", ecolor="black",
                        elinewidth=0.8, capsize=3, capthick=0.8,
                        zorder=3
                    )

    ax.set_ylabel("Execution Success Rate", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.01)
    ax.set_axisbelow(True)        # make grid render beneath
    ax.grid(axis="y", alpha=0.4)  # horizontal grid only
    
    # FULL-BIN gray shading per instance_class
    n_groups = n_rows
    ax.set_xlim(-0.5, n_groups - 0.5)
    for i in range(0, n_groups, 2):  # even-indexed groups
        left = i - 0.5
        right = i + 0.5
        ax.axvspan(left, right, color="#d6d6d6", alpha=0.15, zorder=0)
    
    ax.legend(title="Task Type / Agent", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(args.output_dir, "success_by_object_and_task.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] combined object-class × task-type plot saved to {out_path}")

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

def main():
    args = parse_args()
    tasks = parse_task_config(args.task_config)
    results = parse_results(args, tasks)
    
    rates, df = analyze_results(args, results, args.agent_types)
    plot_overall_success(args, df)
    plot_object_class_success(args, df)
    if args.cascade_error_analysis:
        cascade_failure_analysis(df)

if __name__ == "__main__":  
    main()
