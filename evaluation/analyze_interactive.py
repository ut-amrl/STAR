import argparse
import os
import json
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.container import ErrorbarContainer
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# Shared config (aligned with your baseline)
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
    "random": "Random",
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

# Agent-specific visual style (color only; hatch now encodes variant)
_AGENT_STYLE = {
    "low_level_gt":        {"color": "#4a90d9"},  # lighter blue
    "replan_low_level_gt": {"color": "#ff9d4d"},  # lighter orange
    "high_level_gt":       {"color": "#58c96e"},  # lighter green
    "random":              {"color": "#9e9e9e"},  # lighter gray
}

# Variant style: hatch differentiates noninteractive vs interactive.
_VARIANT_STYLE = {
    "noninteractive": {"hatch": "",    "label": "Non-interactive"},
    "interactive":    {"hatch": "..", "label": "Interactive"},
}

# How much lighter the interactive variant should be (0–1, toward white)
_INTERACTIVE_LIGHTEN = 0.4

def _style_for_agent(agent: str) -> dict:
    return _AGENT_STYLE.get(agent, {"color": "#999999"})

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
    """95% Wilson interval half-width for a binomial proportion."""
    if n <= 0:
        return np.nan
    p = succ / n
    denom = 1.0 + (z**2) / n
    return z * np.sqrt(p * (1 - p) / n + (z**2) / (4 * n * n)) / denom

# ── plotting helpers (same style as baseline) ─────────────────────────────

def _nice_ylim(max_height_with_err: float, pad: float = 0.02, clamp: bool = False):
    """Dynamic ylim; allow >1 unless clamp=True."""
    top = max_height_with_err + pad
    if clamp:
        top = min(1.0, top)
    return (0.0, top)

def _mix(c1, c2, a):
    r1, r2 = np.array(mcolors.to_rgb(c1)), np.array(mcolors.to_rgb(c2))
    return mcolors.to_hex((1 - a) * r1 + a * r2)

def _darken(hex_color, amount=0.25):
    return _mix(hex_color, "#000000", amount)

def _lighten(hex_color, amount=0.22):
    return _mix(hex_color, "#FFFFFF", amount)

def _err_style(ecolor="#333333"):
    return dict(fmt="none", ecolor=ecolor, elinewidth=1.2, capsize=4,
                capthick=1.2, alpha=0.9, zorder=4)

def _round_errcaps(eb: ErrorbarContainer):
    try:
        for lc in eb[2]:
            lc.set_capstyle("round")
            lc.set_joinstyle("round")
        for cap in eb[1]:
            cap.set_solid_capstyle("round")
            cap.set_solid_joinstyle("round")
    except Exception:
        pass

# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="evaluation/sim_outputs/")
    p.add_argument("--output_dir", type=str, default="evaluation/sim_outputs/")
    p.add_argument("--task_config", type=str, default="evaluation/config/tasks_sim_all.txt",)
    p.add_argument("--agent_types", nargs="+",
                   default=["low_level_gt", "replan_low_level_gt"])
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
           'pairs':         {uid, ...}
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
                        "variant": variant,
                        "agent": agent,
                    }
                    row.update(rec)
                    rows.append(row)

    df = pd.DataFrame(rows)

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

    print("\n========== Interactive vs Non-Interactive (paired by UID) ==========")
    for ag, per_task in overall_rates.items():
        print(f"\n=== Agent: {ag} ({_pretty_agent(ag)}) ===")
        for task_t in sorted(per_task.keys(), key=_task_sort_key):
            vdict = per_task[task_t]
            ni = vdict.get("noninteractive", float("nan"))
            iv = vdict.get("interactive", float("nan"))
            print(f"{task_t:22s}  non-int: {ni:.1%}   interactive: {iv:.1%}")

    pair_counts = {tt: len(set(d.keys())) for tt, d in results.items()}
    print("\nPaired UID counts (per base task type):")
    for tt in sorted(pair_counts.keys(), key=_task_sort_key):
        uid_set = set(results[tt].keys())
        print(f"  {tt:22s}: {len(uid_set)}")

    return overall_rates, df

# ────────────────────────────────────────────────────────────────────────────────
# PLOTS
# ────────────────────────────────────────────────────────────────────────────────

def plot_interactive_vs_noninteractive(args, df):
    """
    Colors encode AGENT; hatches encode VARIANT.
    Only one legend is shown: white boxes with hatch patterns for variant.
    """
    import pandas as pd
    plt.rcParams["hatch.linewidth"] = 0.6
    plt.rcParams["hatch.color"] = "black"

    if df.empty or "success" not in df.columns:
        print("[WARN] No data for interactive-vs-noninteractive plot")
        return
    
    # Stats for error bars
    yerr_map = {}
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

    # Aggregate means
    plot_df = (
        df.groupby(["task_type", "variant", "agent"])["success"]
          .mean()
          .reset_index()
          .pivot(index="task_type", columns=["variant", "agent"], values="success")
    )
    if plot_df.empty:
        print("[WARN] Nothing to plot – empty after pivot")
        return

    task_order = [t for t in _TASK_ORDER if t in plot_df.index] + \
                 [t for t in plot_df.index if t not in _TASK_ORDER]
    plot_df = plot_df.reindex(task_order)

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
    width = 0.85 / (n_agents * 2.0)  # two bars per agent
    pair_gap = 0.0
    fig, ax = plt.subplots(figsize=(11, 6))

    max_with_err = 0.0

    for i_task, task in enumerate(tasks):
        for j_agent, ag in enumerate(args.agent_types):
            agent_center = x[i_task] + (j_agent - (n_agents - 1) / 2.0) * (2 * width + pair_gap)

            for var in variants:
                col = (var, ag)
                if col not in plot_df.columns:
                    continue
                val = plot_df.loc[task, col]
                if np.isnan(val):
                    continue

                offset = (-0.5 if var == "noninteractive" else +0.5) * width
                xpos = agent_center + offset

                base_color = _style_for_agent(ag)["color"]
                face_color = base_color if var == "noninteractive" else _lighten(base_color, _INTERACTIVE_LIGHTEN)
                hatch = _VARIANT_STYLE[var]["hatch"]

                bar = ax.bar(
                    xpos, float(val), width=width,
                    color=face_color, edgecolor="black", linewidth=0.7, zorder=2.5
                )[0]
                bar.set_hatch(hatch)
                bar.set_linewidth(0.6)

                yerr = None
                if args.error_bars:
                    yerr = yerr_map.get((task, var, ag))
                    if yerr is not None and np.isfinite(yerr):
                        ec = _darken(base_color, 0.30)
                        eb = ax.errorbar(xpos, float(val), yerr=yerr, **_err_style(ec))
                        _round_errcaps(eb)

                top = float(val) + (float(yerr) if (yerr is not None and np.isfinite(yerr)) else 0.0)
                max_with_err = max(max_with_err, top)

    # Axes / ticks / grid
    ax.set_ylabel("Execution Success Rate", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_task(t) for t in tasks], rotation=45, ha="right")
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.4)
    ax.set_ylim(*_nice_ylim(max_with_err, pad=0.05, clamp=False))

    # Shade alternates
    ax.set_xlim(-0.5, n_tasks - 0.5)
    for i in range(0, len(x), 2):
        ax.axvspan(x[i]-0.5, x[i]+0.5, color="#b7b7b7", alpha=0.12, zorder=0)

    legend_handles = []

    # Agent boxes (solid color, no hatch)
    for ag in args.agent_types:
        legend_handles.append(
            mpatches.Patch(
                facecolor=_style_for_agent(ag)["color"],
                edgecolor="black",
                linewidth=0.7,
                label=_pretty_agent(ag),
            )
        )

    # Variant boxes (white face, hatch)
    legend_handles.extend([
        mpatches.Patch(
            facecolor="white", edgecolor="black", linewidth=0.7,
            hatch=_VARIANT_STYLE["noninteractive"]["hatch"],
            label=_VARIANT_STYLE["noninteractive"]["label"],
        ),
        mpatches.Patch(
            facecolor="white", edgecolor="black", linewidth=0.7,
            hatch=_VARIANT_STYLE["interactive"]["hatch"],
            label=_VARIANT_STYLE["interactive"]["label"],
        ),
    ])

    leg = ax.legend(
        handles=legend_handles,
        fontsize=11,
        loc="upper left", bbox_to_anchor=(1.01, 0.98),
        frameon=True, framealpha=0.9, facecolor="white", edgecolor="lightgray",
        title=None  # no title
    )
    leg.get_title().set_fontweight("medium")

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "success_interactive_vs_noninteractive.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] interactive-vs-noninteractive plot saved to {out_path}")


# ────────────────────────────────────────────────────────────────────────────────
# Cascade failure analysis (variant-aware)
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
    failures = df_sorted[df_sorted[first_flag] == 0]
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
    print("\n[INFO] Pairing summary:")
    for base, info in sorted(pairs_info.items(), key=lambda kv: _task_sort_key(kv[0])):
        print(f"  {base:22s}  non-int: {len(info['noninteractive']) :4d} "
              f"interactive: {len(info['interactive']) :4d} "
              f"pairs: {len(info['pairs']) :4d}")

    results = parse_results(args, pairs_info)
    rates, df = analyze_results(args, results, args.agent_types)

    plot_interactive_vs_noninteractive(args, df)
    # cascade_failure_analysis(df)

if __name__ == "__main__":
    main()
