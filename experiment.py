"""Validate the fixed task and run equal-budget comparisons."""
import argparse
import csv
import json
import platform
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
import matplotlib
from baselines import METHODS, run
from dsl import Program, ACTIONS
from environment import MAZE, START, CHECKPOINTS, GOAL, evaluate, shortest_all_checkpoint_path_steps

ROOT = Path(__file__).parent
# Feasibility reference selected during environment design; never passed to optimizers.
REFERENCE = Program((0, 0, 5, 1, 1, 2, 3, 1, 2))


def validate():
    ref = evaluate(REFERENCE, trace=True)
    constants = []
    for a in range(3):
        p = Program((0,a,0,a,0,a,0,a,a))
        r = evaluate(p)
        constants.append({"action": ACTIONS[a], "reward": r.reward, "success": r.success})
    small = {}
    for k in (1, 2):
        best = float("-inf")
        success = 0
        count = 0
        for components in product(range(18), repeat=k):
            for default in range(3):
                choices = []
                for z in components:
                    choices.extend(divmod(z, 3))
                for _ in range(4-k):
                    choices.extend((0, default))
                choices.append(default)
                r = evaluate(Program(tuple(choices)))
                best = max(best, r.reward)
                success += r.success
                count += 1
        small[str(k)] = {"enumerated": count, "successful": success, "best_reward": best}
    repeat = evaluate(REFERENCE, trace=True)
    deterministic = ref == repeat
    result = {"maze": MAZE, "start": START, "checkpoints": CHECKPOINTS, "goal": GOAL,
              "reference_program": REFERENCE.id, "reference_text": REFERENCE.text(),
              "reference_reward": ref.reward, "reference_success": ref.success,
              "reference_steps": ref.steps, "reference_checkpoints": sum(x["checkpoint_reached"] is not None for x in ref.trace),
              "constant_actions": constants, "small_programs": small, "deterministic": deterministic,
              "shortest_all_checkpoint_path_steps": shortest_all_checkpoint_path_steps()}
    print(json.dumps(result, indent=2))
    if not (ref.success and result["reference_checkpoints"] == 3 and deterministic
            and not any(x["success"] for x in constants)
            and all(small[str(k)]["successful"] == 0 for k in (1, 2))
            and all(small[str(k)]["best_reward"] < ref.reward for k in (1, 2))):
        raise SystemExit("Environment validation failed")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    validation = validate()
    if args.validate:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    evaluations = []
    summaries = []
    for method in METHODS:
        for seed in args.seeds:
            rows, summary = run(method, seed, args.budget)
            evaluations.extend(rows)
            summaries.append(summary)
            print(f"{method} seed={seed} best={summary['final_best_reward']:.3f} success={summary['best_success']} runtime={summary['runtime_seconds']:.2f}s", flush=True)
    def write_csv(name, rows):
        with (args.output / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    write_csv("all_evaluations.csv", evaluations)
    write_csv("per_seed_metrics.csv", summaries)
    aggregate = []
    for method in METHODS:
        subset = [s for s in summaries if s["method"] == method]
        successful = [s["first_success_evaluation"] for s in subset if s["first_success_evaluation"] is not None]
        aggregate.append({"method": method,
                          "mean_final_best_reward": float(np.mean([s["final_best_reward"] for s in subset])),
                          "std_final_best_reward": float(np.std([s["final_best_reward"] for s in subset], ddof=1)) if len(subset)>1 else 0.0,
                          "fraction_runs_success": len(successful)/len(subset),
                          "mean_evaluations_to_first_success_among_successes": float(np.mean(successful)) if successful else "",
                          "successes_over_runs": f"{len(successful)}/{len(subset)}",
                          "mean_runtime_seconds": float(np.mean([s["runtime_seconds"] for s in subset]))})
    write_csv("summary_metrics.csv", aggregate)
    config = {"budget_per_method_seed": args.budget, "seeds": args.seeds, "max_steps": 100,
              "restart_after_failures": 30, "equal_reward_hill_policy": "reject", "maze": MAZE,
              "start": START, "checkpoints": CHECKPOINTS, "goal": GOAL, "validation": validation,
              "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
              "torch": torch.__version__, "matplotlib": matplotlib.__version__, "cuda_used": False, "device": "CPU"}
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
