import argparse
import os
import json
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# Shared config (mirrors your reference)
# ────────────────────────────────────────────────────────────────────────────────

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
    "replan_low_level_gt": "Interleaving Exec.",
    "low_level_gt": "One-shot Exec.",
    "high_level_gt": "high-level",
}

_TASK_DISPLAY_TWO_LINES = {
    "classonly":        "class-\nbased",
    "unambiguous":      "attribute-\nbased",
    "spatial_temporal": "spatial-\ntemporal",
    "spatial":          "spatial",
    "frequency":        "spatial-\nfreqentist",
}

_TASK_DISPLAY = {
    "classonly":        "class-based",
    "unambiguous":      "attribute-based",
    "spatial_temporal": "spatial-temporal",
    "spatial":          "spatial",
    "frequency":        "spatial-freqentist",
}

_COLOR_GROUP = {
    "classonly":        plt.cm.Oranges,
    "unambiguous":      plt.cm.Greens,
    "spatial_temporal": plt.cm.Blues,
    "spatial":          plt.cm.Purples,
    "frequency":        plt.cm.Greys,
}

_TASK_ORDER = [
    "classonly",
    "unambiguous",
    "spatial",
    "spatial_temporal",
    "frequency",
]

def _task_sort_key(t):
    try:
        return _TASK_ORDER.index(t)
    except ValueError:
        return len(_TASK_ORDER) + hash(t)

def _pretty_task(raw: str, oneline: bool = False) -> str:
    display = _TASK_DISPLAY if oneline else _TASK_DISPLAY_TWO_LINES
    for orig, nice in display.items():
        if raw.startswith(orig):
            return f"{nice}{raw[len(orig):]}"
    return raw

def _pretty_agent(raw: str) -> str:
    for orig, nice in _AGENT_DISPLAY.items():
        if raw.startswith(orig):
            return f"{nice}{raw[len(orig):]}"
    return raw

def _strip(record: dict) -> dict:
    return {k: record.get(k) for k in _SUCCESS_FLAGS + _OBJECT_FLAGS}

def _wilson_halfwidth(succ: float, n: int, z: float = 1.96) -> float:
    """95% Wilson interval half-width for a binomial proportion.
    succ: number of successes, n: trials"""
    if n <= 0:
        return np.nan
    p = succ / n
    denom = 1.0 + (z**2) / n
    return z * np.sqrt(p * (1 - p) / n + (z**2) / (4 * n * n)) / denom

# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="evaluation/sim_outputs/")
    p.add_argument("--output_dir", type=str, default="evaluation/sim_outputs/")
    p.add_argument("--task_config", type=str, required=True)
    p.add_argument("--agent_types", nargs="+",
                   default=["low_level_gt", "replan_low_level_gt"])
    # If True, restrict analysis to UIDs that appear in both variants (paired only)
    p.add_argument("--paired_only", action="store_true", default=True)
    p.add_argument("--error_bars", action="store_true",
               help="If set, draw 95% Wilson CI error bars for success rates.")
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────────
# IO helpers
# ────────────────────────────────────────────────────────────────────────────────

def _load_json_safe(path: str):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error in {path}: {e}")
    return None

# ────────────────────────────────────────────────────────────────────────────────
# Parse task_config and build interactive/non-interactive pairing
# ────────────────────────────────────────────────────────────────────────────────

def parse_task_config_pairs(task_config: str) -> Dict[str, Dict[str, set]]:
    """
    Returns:
      {
        base_type: {
           'noninteractive': {uid, ...},
           'interactive':   {uid, ...},
           'pairs':         {uid, ...}     # intersection of both sets
        },
        ...
      }
    """
    out: Dict[str, Dict[str, set]] = defaultdict(lambda: {
        "noninteractive": set(),
        "interactive": set(),
        "pairs": set(),
    })

    with open(task_config, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                tdir, fname = line.split("/", 1)
            except ValueError:
                print(f"[WARN] Cannot parse line: {line}")
                continue

            uid, ext = os.path.splitext(os.path.basename(fname))
            if ext != ".json":
                print(f"[WARN] Expected .json, got {ext} in: {line}")

            if tdir.endswith("_interactive"):
                base = tdir[: -len("_interactive")]
                out[base]["interactive"].add(uid)
            else:
                base = tdir
                out[base]["noninteractive"].add(uid)

    # Build intersections
    for base in list(out.keys()):
        out[base]["pairs"] = out[base]["interactive"].intersection(out[base]["noninteractive"])
    return out

# ────────────────────────────────────────────────────────────────────────────────
# Load results for both variants
# ────────────────────────────────────────────────────────────────────────────────

def parse_results(args, pairs_info: Dict[str, Dict[str, set]]) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    results[base_type][uid][variant][agent] = stripped_json_or_None
    variant ∈ {"interactive","noninteractive"}
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for base_type, info in pairs_info.items():
        # we’ll load both variants for the union of UIDs we care about
        uids = sorted((info["pairs"] if args.paired_only else
                       info["interactive"].union(info["noninteractive"])))

        for uid in uids:
            for variant, tdir in [
                ("noninteractive", f"{base_type}"),
                ("interactive", f"{base_type}_interactive"),
            ]:
                for agent in args.agent_types:
                    fname = f"results_{agent}_{uid}.json"
                    path = os.path.join(args.data_dir, tdir, uid, fname)
                    raw = _load_json_safe(path)
                    results[base_type][uid][variant][agent] = _strip(raw) if raw else None
    return results

# ────────────────────────────────────────────────────────────────────────────────
# Build DataFrame and compute rates
# ────────────────────────────────────────────────────────────────────────────────

def analyze_results(args,
                    results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
                    agent_types: List[str]):
    import pandas as pd

    rows = []
    for base_type, per_uid in results.items():
        for uid, per_variant in per_uid.items():
            for variant in ("noninteractive", "interactive"):
                for agent in agent_types:
                    rec = per_variant.get(variant, {}).get(agent)
                    if rec is None:
                        continue
                    row = {
                        "task_type": base_type,
                        "uid": uid,
                        "variant": variant,   # ← NEW
                        "agent": agent,
                    }
                    row.update(rec)
                    rows.append(row)

    df = pd.DataFrame(rows)

    # Compute per-agent, per-base task_type, per-variant success means over the paired set
    # (We assume the canonical 'success' column exists in records; if not, you can switch to another flag)
    def _rates(df_in):
        rates = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for agent in agent_types:
            sub_a = df_in[df_in["agent"] == agent]
            for task_type, g_tt in sub_a.groupby("task_type"):
                for variant, g_var in g_tt.groupby("variant"):
                    if "success" in g_var.columns:
                        denom = g_var["success"].notna().sum()
                        if denom:
                            rates[agent][task_type][variant] = g_var["success"].mean()
        return rates

    overall_rates = _rates(df)

    # Pretty print summary
    print("\n========== Interactive vs Non-Interactive (paired by UID) ==========")
    for ag, per_task in overall_rates.items():
        print(f"\n=== Agent: {ag} ({_pretty_agent(ag)}) ===")
        for task_t in sorted(per_task.keys(), key=_task_sort_key):
            vdict = per_task[task_t]
            ni = vdict.get("noninteractive", float("nan"))
            iv = vdict.get("interactive", float("nan"))
            print(f"{task_t:22s}  non-int: {ni:.1%}   interactive: {iv:.1%}")

    # Also show how many UIDs per base type used in pairing
    pair_counts = {
        tt: len(set(d.keys()))  # #UID rows in df for this task_type & variant will be 2× per UID; here we just count UIDs
        for tt, d in results.items()
    }
    print("\nPaired UID counts (per base task type):")
    for tt in sorted(pair_counts.keys(), key=_task_sort_key):
        # Count true pair UIDs that showed up at least once
        uid_set = set(results[tt].keys())
        # len(uid_set) is the number of UIDs we attempted to load (paired_only or union)
        print(f"  {tt:22s}: {len(uid_set)}")

    return overall_rates, df

# ────────────────────────────────────────────────────────────────────────────────
# PLOTS
# ────────────────────────────────────────────────────────────────────────────────

def plot_interactive_vs_noninteractive(args, df):
    """
    X-axis: base task_type.
    Within each task_type group: for each agent, show two bars side-by-side:
        [non-interactive | interactive]
    Color encodes AGENT (via shade inside the task-type colormap), hatch encodes VARIANT.
    Legend: single box (Agent colors + Variant hatches).
    """
    import pandas as pd
    if df.empty or "success" not in df.columns:
        print("[WARN] No data for interactive-vs-noninteractive plot")
        return
    
    # Precompute counts & Wilson CI half-widths per (task_type, variant, agent)
    stats_gb = (
        df[["task_type", "variant", "agent", "success"]]
        .dropna(subset=["success"])
        .groupby(["task_type", "variant", "agent"])
    )
    _stats = stats_gb.agg(mean=("success", "mean"),
                        n=("success", "count"),
                        s=("success", "sum")).reset_index()
    yerr_map = {
        (r["task_type"], r["variant"], r["agent"]): _wilson_halfwidth(r["s"], int(r["n"]))
        for _, r in _stats.iterrows()
    }

    # Aggregate mean success by (task_type, variant, agent)
    plot_df = (
        df.groupby(["task_type", "variant", "agent"])["success"]
          .mean()
          .reset_index()
          .pivot(index="task_type", columns=["variant", "agent"], values="success")
    )
    if plot_df.empty:
        print("[WARN] Nothing to plot – empty after pivot")
        return

    # Task order
    task_order = [t for t in _TASK_ORDER if t in plot_df.index] + \
                 [t for t in plot_df.index if t not in _TASK_ORDER]
    plot_df = plot_df.reindex(task_order)

    # Column order: for each agent → [noninteractive, interactive]
    variants = ["noninteractive", "interactive"]
    flat_cols = []
    for ag in args.agent_types:
        for var in variants:
            col = (var, ag)
            if col in plot_df.columns:
                flat_cols.append(col)
    if not flat_cols:
        print("[WARN] No matching (variant, agent) columns to plot")
        return
    plot_df = plot_df.reindex(columns=pd.MultiIndex.from_tuples(flat_cols))

    tasks = plot_df.index.tolist()
    n_tasks = len(tasks)
    n_agents = len(args.agent_types)

    x = np.arange(n_tasks)
    # width chosen so group (per agent) has two bars: NI and I
    width = 0.85 / (n_agents * 2.0)  # two bars per agent
    pair_gap = 0.0                   # set >0 to open space between NI and I
    fig, ax = plt.subplots(figsize=(11, 6))

    # Shades per agent (light→dark) from the task colormap
    shade_points = np.linspace(0.25, 0.85, n_agents)

    for i_task, task in enumerate(tasks):
        cmap = _COLOR_GROUP.get(task, plt.cm.tab10)
        for j_agent, (ag, shade) in enumerate(zip(args.agent_types, shade_points)):
            # center of this agent's 2-bar pair within the task group
            agent_center = x[i_task] + (j_agent - (n_agents - 1) / 2.0) * (2 * width + pair_gap)

            # Two variants: noninteractive (left), interactive (right)
            for k, var in enumerate(variants):
                col = (var, ag)
                if col not in plot_df.columns:
                    continue
                val = plot_df.loc[task, col]
                if pd.isna(val):
                    continue  # skip missing

                # offset within the agent pair: -0.5*width (NI) and +0.5*width (I)
                offset = (-0.5 if var == "noninteractive" else +0.5) * width
                xpos = agent_center + offset

                bar = ax.bar(
                    xpos,
                    val,
                    width=width,
                    color=cmap(shade),
                    edgecolor="black",
                    linewidth=0.7,
                )[0]
                if var == "interactive":
                    bar.set_hatch("//")

                if args.error_bars:
                    yerr = yerr_map.get((task, var, ag))
                    if yerr is not None and np.isfinite(yerr):
                        ax.errorbar(
                            xpos, val, yerr=yerr,
                            fmt="none", ecolor="black",
                            elinewidth=0.8, capsize=3, capthick=0.8,
                            zorder=3
                        )

    # x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_task(t) for t in tasks], rotation=45, ha="right")

    # y-axis & grid
    ax.set_ylabel("Execution Success Rate", fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.4)

    # Group shading for readability
    ax.set_xlim(-0.5, n_tasks - 0.5)  # ensures full bin is visible
    for i in range(0, len(x), 2):      # shade every other class
        left = x[i] - 0.5
        right = x[i] + 0.5
        ax.axvspan(left, right, color="#808080", alpha=0.15, zorder=0)

    # ── Single legend box: Agent (color) + Variant (hatch) ─────────────────
    if tasks:
        # Agent handles (color from first task's cmap)
        cmap0 = _COLOR_GROUP.get(tasks[0], plt.cm.tab10)
        agent_handles = [
            mpatches.Patch(facecolor=cmap0(shade), edgecolor="black",
                           label=_pretty_agent(ag), linewidth=0.7)
            for ag, shade in zip(args.agent_types, shade_points)
        ]
        # Variant handles (hatch)
        variant_handles = [
            mpatches.Patch(facecolor="white", edgecolor="black",
                           hatch="", label="Non-interactive", linewidth=0.7),
            mpatches.Patch(facecolor="white", edgecolor="black",
                           hatch="//", label="Interactive", linewidth=0.7),
        ]
        handles = agent_handles + variant_handles
        ax.legend(
            handles=handles,
            title="Legend — color: Agent, hatch: Variant",
            bbox_to_anchor=(1.04, 1), loc="upper left"
        )

    ax.set_title("Success by Task Type — Agent-wise: Non-Interactive vs Interactive")

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "success_interactive_vs_noninteractive.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] interactive-vs-noninteractive plot saved to {out_path}")


# ────────────────────────────────────────────────────────────────────────────────
# Existing cascade failure analysis (reused; runs on the full df with variant column)
# ────────────────────────────────────────────────────────────────────────────────

def cascade_failure_analysis(df,
                             success_flags=_SUCCESS_FLAGS,
                             group_keys=["task_type", "uid", "variant", "agent"]):
    import pandas as pd
    print("\n=== Cascade Failure Analysis (variant-aware) ===")
    missing = [flag for flag in success_flags if flag not in df.columns]
    if missing:
        print(f"[WARN] Missing columns in dataframe: {missing}")
        return

    df_sorted = df.sort_values(group_keys)

    first_flag = success_flags[0]
    print(f"\n--- FAILED at '{first_flag}' (first stage) ---")
    mask = (df_sorted[first_flag] == 0)
    failures = df_sorted[mask]
    if failures.empty:
        print("(none)")
    else:
        print(failures[group_keys + [first_flag]])

    for i in range(1, len(success_flags)):
        prev_flag = success_flags[i-1]
        curr_flag = success_flags[i]
        print(f"\n--- SUCCEEDED at '{prev_flag}' but FAILED at '{curr_flag}' ---")
        mask = (df_sorted[prev_flag] == 1) & (df_sorted[curr_flag] == 0)
        failures = df_sorted[mask]
        if failures.empty:
            print("(none)")
        else:
            print(failures[group_keys + [prev_flag, curr_flag]])

    print("\n=== End Cascade Failure Analysis ===")

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    pairs_info = parse_task_config_pairs(args.task_config)
    # quick summary
    print("\n[INFO] Pairing summary:")
    for base, info in sorted(pairs_info.items(), key=lambda kv: _task_sort_key(kv[0])):
        print(f"  {base:22s}  non-int: {len(info['noninteractive']) :4d} "
              f"interactive: {len(info['interactive']) :4d} "
              f"pairs: {len(info['pairs']) :4d}")

    results = parse_results(args, pairs_info)
    rates, df = analyze_results(args, results, args.agent_types)

    # Plot the new interactive vs noninteractive comparison
    plot_interactive_vs_noninteractive(args, df)

    # (Optional) you can still produce your original overall plots if desired,
    # reusing your previous functions; here we just run the cascade analysis:
    cascade_failure_analysis(df)

if __name__ == "__main__":
    main()
