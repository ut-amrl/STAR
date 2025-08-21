#!/usr/bin/env python3
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Print missing tool-call files (reasoning_toolcalls_{agent}_{uid}.json) per agent."
    )
    p.add_argument("--data_dir", type=str, default="evaluation/sim_outputs/",
                   help="Root directory containing task_type/uid subfolders.")
    p.add_argument("--task_config", type=str, default="evaluation/config/tasks_sim_all.txt")
    p.add_argument("--agent_types", nargs="+",
                   default=["low_level_gt", "replan_low_level_gt"],
                   help="Agents to check.")
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

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
            if ext and ext != ".json":
                print(f"[WARN] Expected .json, got {ext} in: {line}")
            tasks[task_type].append(uid)
    return dict(tasks)

def toolcalls_path(data_dir: str, task_type: str, uid: str, agent: str) -> str:
    """reasoning_toolcalls_{agent}_{uid}.json under data_dir/task_type/uid/"""
    fname = f"reasoning_toolcalls_{agent}_{uid}.json"
    return os.path.join(data_dir, task_type, uid, fname)

# ────────────────────────────────────────────────────────────────────────────────
# Core
# ────────────────────────────────────────────────────────────────────────────────

def find_missing_toolcalls(tasks: Dict[str, List[str]],
                           agents: List[str],
                           data_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    """Return {agent: [(task_type, uid), ...] missing}."""
    missing: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for agent in agents:
        for task_type, uids in tasks.items():
            for uid in uids:
                path = toolcalls_path(data_dir, task_type, uid, agent)
                if not os.path.isfile(path):
                    missing[agent].append((task_type, uid))
    return missing

def print_missing_report(tasks: Dict[str, List[str]],
                         missing: Dict[str, List[Tuple[str, str]]],
                         agents: List[str]):
    expected_total = sum(len(uids) for uids in tasks.values())
    for agent in agents:
        miss_list = sorted(missing.get(agent, []))
        print(f"\n===== Agent: {agent} =====")
        print(f"Expected tasks: {expected_total}   Missing: {len(miss_list)}")
        if not miss_list:
            print("(no missing files)")
            continue
        # Print in the same style as task_config (task_type/uid.json)
        for task_type, uid in miss_list:
            print(f"{task_type}/{uid}.json")

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    tasks = parse_task_config(args.task_config)
    missing = find_missing_toolcalls(tasks, args.agent_types, args.data_dir)
    print_missing_report(tasks, missing, args.agent_types)

if __name__ == "__main__":
    main()
