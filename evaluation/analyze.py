import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Read evaluation results.")
    parser.add_argument(
        "--result_file",
        type=str,
        default="evaluation/outputs/results.json",
        help="Path to the results JSON file (default: evaluation/outputs/results.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/outputs/",
        help="Path to save the output plot (default: evaluation/outputs/)"
    )
    return parser.parse_args()

def load_results(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def compute_success_rates(results):
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # task_type → instance_class → [success_count, total_count]
    
    for task_type, records in results.get("results", {}).items():
        for r in records:
            cls = r.get("instance_class", "unknown")
            success = r["success"]
            stats[task_type][cls][1] += 1
            if success:
                stats[task_type][cls][0] += 1
    return stats

def plot_success_rates(args, stats):
    task_types = sorted(stats.keys())
    all_classes = sorted({cls for t in stats.values() for cls in t})

    x = np.arange(len(all_classes))  # the label locations
    width = 0.8 / len(task_types)    # bar width, adjusted for group spacing

    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, task_type in enumerate(task_types):
        heights = []
        for cls in all_classes:
            success, total = stats[task_type].get(cls, [0, 0])
            rate = success / total if total > 0 else 0.0
            heights.append(rate)
        ax.bar(x + i * width, heights, width, label=task_type)
    
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate by Instance Class and Task Type")
    ax.set_xticks(x + width * (len(task_types) - 1) / 2)
    ax.set_xticklabels(all_classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Task Type")

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "success_rates.png")
    plt.savefig(output_path)
    print(f"✅ Plot saved to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    results_data = load_results(args.result_file)
    stats = compute_success_rates(results_data)
    plot_success_rates(args, stats)
    
    # Just print a quick summary to check
    for task_type, result_list in results_data.get("results", {}).items():
        total = len(result_list)
        success = sum(r["success"] for r in result_list)
        print(f"{task_type:<20}: {success}/{total} succeeded ({100 * success / total:.1f}%)")
