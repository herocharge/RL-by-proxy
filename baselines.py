"""The three fixed-budget program optimizers."""
import random
import time
import numpy as np
import torch
from dsl import random_program, mutate
from environment import evaluate
from reinforce import Reinforce

METHODS = ("random", "hill", "reinforce")


def run(method: str, seed: int, budget: int, restart_after: int = 30):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = random.Random(seed)
    rows = []
    best_program = None
    best_reward = float("-inf")
    best_success = False
    first_success = None
    current_program = None
    current_reward = float("-inf")
    failures = 0
    learner = Reinforce() if method == "reinforce" else None
    start = time.perf_counter()
    for i in range(1, budget + 1):
        if method == "random":
            program = random_program(rng)
            result = evaluate(program)
        elif method == "hill":
            if current_program is None or failures >= restart_after:
                program = random_program(rng)
                result = evaluate(program)
                current_program, current_reward, failures = program, result.reward, 0
            else:
                program = mutate(current_program, rng)
                result = evaluate(program)
                if result.reward > current_reward:
                    current_program, current_reward, failures = program, result.reward, 0
                else:
                    failures += 1  # equal rewards rejected
        elif method == "reinforce":
            program, result = learner.sample_and_update(evaluate)
        else:
            raise ValueError(method)
        if result.reward > best_reward or (result.reward == best_reward and result.success and not best_success):
            best_program, best_reward, best_success = program, result.reward, result.success
        if result.success and first_success is None:
            first_success = i
        rows.append({"method": method, "seed": seed, "evaluation": i, "program_id": program.id,
                     "choices": program.id, "reward": result.reward, "success": result.success,
                     "best_so_far": best_reward, "best_success": best_success})
    elapsed = time.perf_counter() - start
    summary = {"method": method, "seed": seed, "budget": budget, "final_best_reward": best_reward,
               "best_success": best_success, "first_success_evaluation": first_success,
               "runtime_seconds": elapsed, "best_program_id": best_program.id,
               "best_program_text": best_program.text()}
    return rows, summary
