"""One-time exhaustive feasibility and difficulty check for the frozen DSL/maze."""
import argparse
import json
import time
from itertools import product
from pathlib import Path

from dsl import Program
from environment import MAZE, START, CHECKPOINTS, GOAL, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/full_space_validation.json"))
    args = parser.parse_args()
    start = time.perf_counter()
    count = successful = 0
    best_reward = float("-inf")
    best_program = None
    for pairs in product(range(18), repeat=4):
        for default in range(3):
            choices = []
            for pair in pairs:
                choices.extend(divmod(pair, 3))
            choices.append(default)
            program = Program(tuple(choices))
            result = evaluate(program)
            count += 1
            successful += result.success
            if result.reward > best_reward:
                best_reward, best_program = result.reward, program
    report = {"maze": MAZE, "start": START, "checkpoints": CHECKPOINTS, "goal": GOAL,
              "programs_enumerated": count, "goal_reaching_programs": successful,
              "best_reward": best_reward, "example_best_program": best_program.id,
              "runtime_seconds": time.perf_counter() - start}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
