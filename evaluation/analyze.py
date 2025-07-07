import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

EXPECTED_TASK_TYPES = [
    # "unambiguous",
    # "spatial",
    # "spatial_temporal"
    # "unambiguous_wp_only",
    # "spatial_wp_only",
    "spatial_temporal_wp_only",
    "spatial_temporal_recaption_wp_only"
]
TASK_TYPE_ORDER = [
    # "unambiguous",
    # "unambiguous_wp_only",
    # "spatial",
    # "spatial_wp_only",
    # "spatial_temporal", 
    "spatial_temporal_wp_only", 
    "spatial_temporal_recaption_wp_only"
]
TASK_DISPLAY_NAMES = {
    "unambiguous": "attribute-based",
    "spatial": "spatial",  # leave unchanged
    "spatial_temporal": "temporal",  # if used in future
    "unambiguous_wp_only": "attribute-based",
    "spatial_wp_only": "spatial",
    "spatial_temporal_wp_only": "temporal",
    "spatial_temporal_recaption_wp_only": "temporal (recaption)"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Read evaluation results.")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="evaluation/outputs/",
        help="Directory containing results_{task_type}.json files (default: evaluation/outputs/)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/outputs/",
        help="Path to save the output plot (default: evaluation/outputs/)"
    )
    return parser.parse_args()


def load_results_from_dir(result_dir, task_types):
    result_dir = Path(result_dir)
    results = {}
    for task_type in task_types:
        file_path = result_dir / f"results_{task_type}.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                results[task_type] = json.load(f)
        else:
            print(f"⚠️  Skipping missing file: {file_path}")
    return results


def get_present_metric_keys(results):
    """Return a list of metric keys that are present in at least one result entry."""
    metric_keys = set()
    for task_results in results.values():
        for r in task_results:
            for key in ("success", "mem_success"):
                if key in r:
                    metric_keys.add(key)
        # Break early if both keys are found
        if "success" in metric_keys and "mem_success" in metric_keys:
            break
    return list(metric_keys)


def compute_success_rates(results, key: str):
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # task_type → instance_class → [success_count, total_count]
    for task_type, task_results in results.items():
        for r in task_results:
            if key not in r:
                continue
            cls = r.get("instance_class", "unknown")
            success = r[key]
            stats[task_type][cls][1] += 1
            if success:
                stats[task_type][cls][0] += 1
    return stats


def plot_combined_success_rates(args, stats_dict, metric_keys):
    task_types = [t for t in TASK_TYPE_ORDER if any(t in stats for stats in stats_dict.values())]
    all_classes = sorted({cls for stats in stats_dict.values() for task_stats in stats.values() for cls in task_stats})
    x = np.arange(len(all_classes))
    width = 0.8 / len(task_types)

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(9 * len(metric_keys), 6), sharex=True, sharey=True)
    if len(metric_keys) == 1:
        axes = [axes]  # ensure iterable

    color_groups = {
        "unambiguous": plt.cm.Greens,
        "spatial_temporal": plt.cm.Blues,
        "spatial": plt.cm.Purples
    }

    # Prepare color assignment
    type_to_color = {}
    grouped = defaultdict(list)
    for task_type in task_types:
        for prefix in color_groups:
            if task_type.startswith(prefix):
                grouped[prefix].append(task_type)
                break

    for prefix, type_list in grouped.items():
        cmap = color_groups[prefix]
        for i, task_type in enumerate(type_list):
            color = cmap(0.3 + 0.5 * i / max(len(type_list) - 1, 1))
            type_to_color[task_type] = color

    for ax, key in zip(axes, metric_keys):
        stats = stats_dict[key]
        for i, task_type in enumerate(task_types):
            heights = []
            for cls in all_classes:
                success, total = stats[task_type].get(cls, [0, 0])
                rate = success / total if total > 0 else 0.0
                heights.append(rate)
            color = type_to_color.get(task_type, "gray")
            ax.bar(
                x + i * width,
                heights,
                width,
                label=TASK_DISPLAY_NAMES.get(task_type, task_type),
                color=color,
                edgecolor="black",
                linewidth=0.5
            )
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)
            ax.set_axisbelow(True)

        # Labeling
        title = {
            "success": "Memory Recall Accuracy",
            "mem_success": "Memory Recall Accuracy"
        }.get(key, key.replace("_", " ").title())

        ylabel = {
            "success": "Memory Recall Accuracy",
            "mem_success": "Memory Recall Accuracy"
        }.get(key, "Success Rate")

        ax.set_title(title, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=14)

    # Shared axis and formatting
    center_positions = x + width * (len(task_types) - 1) / 2
    for ax in axes:
        ax.set_xticks(center_positions)
        ax.set_xticklabels(all_classes, rotation=45, ha="right", fontsize=12)
        ax.legend(title="Task Type")

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = Path(args.output_dir) / "accuracy.png"
    plt.savefig(out_path)
    print(f"✅ Combined plot saved to {out_path}")


def print_summary(results_data, key, label):
    print(f"Summary of {label} Results:")
    for task_type, task_results in results_data.items():
        valid = [r for r in task_results if key in r]
        total = len(valid)
        success = sum(r[key] for r in valid)
        rate = success / total if total > 0 else 0.0
        stderr = (rate * (1 - rate) / total) ** 0.5 if total > 0 else 0.0
        ci95 = 1.96 * stderr
        print(f"{task_type:<20}: {success}/{total} succeeded ({rate*100:.1f}% ± {ci95*100:.1f}%)")


if __name__ == "__main__":
    args = parse_args()
    all_task_types = EXPECTED_TASK_TYPES.copy()
    results_data = load_results_from_dir(args.result_dir, all_task_types)
    present_keys = get_present_metric_keys(results_data)
    preferred_order = ["mem_success", "success"]
    present_keys = [k for k in preferred_order if k in present_keys]

    if not present_keys:
        print("❌ No valid metric keys found (expected 'success' or 'mem_success')")
        exit(1)

    stats_dict = {
        key: compute_success_rates(results_data, key)
        for key in present_keys
    }

    plot_combined_success_rates(args, stats_dict, present_keys)

    if "success" in present_keys:
        print_summary(results_data, "success", "Execution")
    if "mem_success" in present_keys:
        print_summary(results_data, "mem_success", "Memory Recall")
