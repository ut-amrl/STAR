import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

EXPECTED_TASK_TYPES = [
    "unambiguous", 
    # "spatial", 
    # "spatial_temporal"
]
TASK_TYPE_ORDER = [
    "unambiguous", 
    # "unambiguous_wp_only",
    # "spatial", 
    # "spatial_wp_only",
    # "spatial_temporal", "spatial_temporal_wp_only", "spatial_temporal_recaption_wp_only"
]
ALL_TASK_TYPES = []

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

def load_results_from_dir(result_dir):
    result_dir = Path(result_dir)
    results = {}
    for task_type in ALL_TASK_TYPES:
        file_path = os.path.join(result_dir, f"results_{task_type}.json")
        file_path = Path(file_path)
        if file_path.exists():
            with open(file_path, "r") as f:
                results[task_type] = json.load(f)
        else:
            print(f"⚠️  Skipping missing file: {file_path}")
    return results

def compute_success_rates(results, key:str = "success"):
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

def plot_success_rates(args, stats, type: str):
    task_types = [t for t in TASK_TYPE_ORDER if t in stats]
    all_classes = sorted({cls for t in stats.values() for cls in t})
    
    x = np.arange(len(all_classes))
    width = 0.8 / len(task_types)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_axisbelow(True)
    
    # Group task_types by prefix
    color_groups = {
        "unambiguous": plt.cm.Greens,
        "spatial_temporal": plt.cm.Blues,
        "spatial": plt.cm.Purples
    }

    # Assign colors based on prefix families
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
            color = cmap(0.3 + 0.5 * i / max(len(type_list) - 1, 1))  # spaced shade
            type_to_color[task_type] = color

    for i, task_type in enumerate(task_types):
        heights = []
        for cls in all_classes:
            success, total = stats[task_type].get(cls, [0, 0])
            rate = success / total if total > 0 else 0.0
            heights.append(rate)
        color = type_to_color.get(task_type, "gray")  # fallback
        ax.bar(x + i * width, heights, width, label=task_type, color=color, edgecolor="black", linewidth=0.5)

    if type == "retrieval":
        ax.set_ylabel("Retrieval Accuracy", fontsize=18)
        ax.set_title("Retrieval Accuracy by Instance Class and Task Type")
    elif type == "execution":
        ax.set_ylabel("Execution Success Rate", fontsize=18)
        ax.set_title("Execution Success Rate by Instance Class and Task Type")
    else:
        raise ValueError(f"Unknown type: {type}. Expected 'retrieval' or 'execution'.")
    ax.set_xticks(x + width * (len(task_types) - 1) / 2)
    ax.set_xticklabels(all_classes, rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Task Type")

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{type}_accuracy.png")
    plt.savefig(output_path)
    print(f"✅ Plot saved to {output_path}")

if __name__ == "__main__":
    for task_type in EXPECTED_TASK_TYPES:
        ALL_TASK_TYPES.append(task_type)
        # ALL_TASK_TYPES.append(f"{task_type}_wp_only")
        # ALL_TASK_TYPES.append(f"{task_type}_recaption_wp_only")
    
    args = parse_args()
    results_data = load_results_from_dir(args.result_dir)
    
    stats = compute_success_rates(results_data, key="success")
    plot_success_rates(args, stats, type="execution")
    stats = compute_success_rates(results_data, key="mem_success")
    plot_success_rates(args, stats, type="retrieval")
    
    # Just print a quick summary to check
    print("Summary of Execution Results:")
    for task_type, task_results in results_data.items():
        total = len(task_results)
        success = sum(r["success"] for r in task_results)
        rate = success / total if total > 0 else 0.0
        stderr = (rate * (1 - rate) / total) ** 0.5 if total > 0 else 0.0
        ci95 = 1.96 * stderr
        print(f"{task_type:<20}: {success}/{total} succeeded "
            f"({rate*100:.1f}% ± {ci95*100:.1f}%)")
        
    print("Summary of Memory Recall Results:")
    # Just print a quick summary to check
    for task_type, task_results in results_data.items():
        total = len(task_results)
        success = sum(r["mem_success"] for r in task_results)
        rate = success / total if total > 0 else 0.0
        stderr = (rate * (1 - rate) / total) ** 0.5 if total > 0 else 0.0
        ci95 = 1.96 * stderr
        print(f"{task_type:<20}: {success}/{total} succeeded "
            f"({rate*100:.1f}% ± {ci95*100:.1f}%)")
