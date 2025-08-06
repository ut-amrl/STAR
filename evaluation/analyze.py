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

_TASK_DISPLAY = {
    "unambiguous":      "attribute-based",
    "spatial_temporal": "temporal",
    "frequency":        "freqentist",
    # add more renames here if needed
}

_COLOR_GROUP = {
    "unambiguous":  plt.cm.Greens,
    "spatial_temporal": plt.cm.Blues,
    "spatial":          plt.cm.Purples,
    "frequency":        plt.cm.Greys,
}

# desired left-to-right order in every grouped plot  (raw names!)
_TASK_ORDER = [
    "unambiguous",       # → “attribute-based”
    "spatial_temporal",  # → “temporal”
    "frequency",         # → “freqencist”
]

def _task_sort_key(t):
    try:
        return _TASK_ORDER.index(t)
    except ValueError:
        return len(_TASK_ORDER) + hash(t)

def _pretty_task(raw: str) -> str:
    """Human-friendly label for legends / tick-labels."""
    for orig, nice in _TASK_DISPLAY.items():
        if raw.startswith(orig):
            return f"{nice}{raw[len(orig):]}"     # keep any suffix
    return raw

def _strip(record: dict) -> dict:
    """Return only the fields listed in _SUCCESS_FLAGS and _OBJECT_FLAGS (missing keys → None)."""
    return {k: record.get(k) for k in _SUCCESS_FLAGS + _OBJECT_FLAGS}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="evaluation/sim_outputs/")
    parser.add_argument("--output_dir", type=str, default="evaluation/sim_outputs/")
    parser.add_argument("--task_config", type=str, default="evaluation/config/tasks_sim.txt")
    parser.add_argument("--agent_types", nargs="+", default=["low_level_gt", "high_level_gt"])
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
        tasks = sorted(agg.index.tolist(), key=_task_sort_key)
        n_tasks = len(tasks)
        n_agents = len(args.agent_types)
        x = np.arange(n_tasks)
        width = 0.8 / n_agents

        fig, ax = plt.subplots(figsize=(11, 6))

        shade_points = np.linspace(0.45, 0.75, n_agents)    # same as helper

        for i_agent, (ag, shade) in enumerate(zip(args.agent_types, shade_points)):
            for i_task, task in enumerate(tasks):
                cmap  = _COLOR_GROUP.get(task.replace("-", "_"), plt.cm.tab10)
                colour = cmap(shade)
                ax.bar(
                    x[i_task] + (i_agent - (n_agents-1)/2) * width,
                    agg.loc[task, ag],
                    width=width,
                    color=colour,
                    edgecolor="black",
                    linewidth=0.7,
                )

        # x-tick labels
        ax.set_xticks(x)
        ax.set_xticklabels([_pretty_task(t) for t in tasks], rotation=45, ha="right")

        # y-axis & grid
        ax.set_ylabel("Execution Success Rate", fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.4)

        # legend – one patch per agent (colour sampled from first task)
        legend_handles = [
            mpatches.Patch(
                facecolor=_COLOR_GROUP.get(tasks[0].replace("-", "_"), plt.cm.tab10)(shade_points[i]),
                edgecolor="black",
                label=ag,
                linewidth=0.7,
            )
            for i, ag in enumerate(args.agent_types)
        ]
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

    # ── 1 aggregate success per (instance_class, task_type, agent) ─────────
    plot_df = (
        df.groupby(["instance_class", "task_type", "agent"])["success"]
          .mean()
          .reset_index()
          .pivot(index="instance_class",
                 columns=["task_type", "agent"],
                 values="success")
    )
    if plot_df.empty:
        print("[WARN] Nothing to plot – empty after pivot")
        return

    # ── 2 build display names & colours in one pass ───────────────────────
    flat_cols, colours = [], []
    unique_tasks = {t for t, _ in plot_df.columns}
    ordered_tasks = [t for t in _TASK_ORDER if t in unique_tasks] + \
                    [t for t in unique_tasks if t not in _TASK_ORDER]

    for task in ordered_tasks:
        cmap   = _COLOR_GROUP.get(task, plt.cm.tab10)       # fallback palette
        agents = args.agent_types
        # pick evenly spaced shades   light → dark
        shade_points = np.linspace(0.45, 0.75, len(agents))
        for shade, ag in zip(shade_points, agents):
            flat_cols.append((task, ag))
            colours.append(cmap(shade))

    # re-index columns so the dataframe matches the colour list
    plot_df = plot_df.reindex(columns=pd.MultiIndex.from_tuples(flat_cols))

    # flatten column names for Matplotlib (“temporal\nlow_level”, …)
    plot_df.columns = [f"{_pretty_task(t)}\n{ag}" for t, ag in plot_df.columns]

    # ── 3 draw ─────────────────────────────────────────────────────────────
    ax = plot_df.plot(
        kind="bar",
        rot=45,
        figsize=(11, 6),
        color=colours,
        edgecolor="black",
        linewidth=0.7,
        width=0.7,
    )
    ax.set_ylabel("Execution Success Rate", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.01)
    ax.set_axisbelow(True)        # make grid render beneath
    ax.grid(axis="y", alpha=0.4)  # horizontal grid only
    # ax.set_title("Execution Success Rate")
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
    cascade_failure_analysis(df)

if __name__ == "__main__":  
    main()