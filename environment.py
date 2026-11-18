"""Deterministic 8x8 Karel-lite maze and shared program evaluator."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from dsl import Program

MAZE = (
    "########",
    "#...#..#",
    "#......#",
    "#.#.#..#",
    "#.#....#",
    "#....###",
    "#..#.###",
    "########",
)
START = (1, 1, 1)  # row, column, direction: 0=N, 1=E, 2=S, 3=W
CHECKPOINTS = ((2, 1), (5, 1), (5, 2))
GOAL = (6, 4)
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
MAX_STEPS = 100


@dataclass
class Result:
    reward: float
    success: bool
    steps: int
    trace: list[dict]
    activation_counts: list[int]


def clear(row: int, col: int, direction: int) -> bool:
    dr, dc = DIRECTIONS[direction]
    nr, nc = row + dr, col + dc
    return 0 <= nr < 8 and 0 <= nc < 8 and MAZE[nr][nc] != "#"


def observe(row: int, col: int, direction: int) -> tuple[bool, bool, bool]:
    return (clear(row, col, direction), clear(row, col, (direction - 1) % 4), clear(row, col, (direction + 1) % 4))


def evaluate(program: Program, trace: bool = False, max_steps: int = MAX_STEPS) -> Result:
    row, col, direction = START
    visited: set[tuple[int, int]] = set()
    records: list[dict] = []
    counts = [0] * 5
    total = 0.0
    for t in range(max_steps):
        obs = observe(row, col, direction)
        action, fired = program.act(obs)
        counts[fired] += 1
        if trace:
            before = {"row": row, "col": col, "direction": direction, "checkpoints_visited": sorted([list(p) for p in visited])}
        if action == 0 and obs[0]:
            dr, dc = DIRECTIONS[direction]
            row, col = row + dr, col + dc
        elif action == 1:
            direction = (direction - 1) % 4
        elif action == 2:
            direction = (direction + 1) % 4
        new_checkpoint = (row, col) in CHECKPOINTS and (row, col) not in visited
        if new_checkpoint:
            visited.add((row, col))
        success = (row, col) == GOAL
        reward = -0.002 + (0.2 if new_checkpoint else 0.0) + (1.0 if success else 0.0)
        total += reward
        if trace:
            records.append({"program_id": program.id, "timestep": t, "full_environment_state": before,
                            "observation": dict(zip(("front_clear", "left_clear", "right_clear"), obs)),
                            "rule_fired": fired + 1 if fired < 4 else "default", "environment_action": ("move", "turn_left", "turn_right")[action],
                            "immediate_reward": reward, "next_environment_state": {"row": row, "col": col, "direction": direction,
                            "checkpoints_visited": sorted([list(p) for p in visited])},
                            "checkpoint_reached": list((row, col)) if new_checkpoint else None,
                            "termination_reason": "goal" if success else ("step_limit" if t == max_steps - 1 else None)})
        if success:
            return Result(round(total, 12), True, t + 1, records, counts)
    return Result(round(total, 12), False, max_steps, records, counts)


def shortest_all_checkpoint_path_steps() -> int:
    """Unrestricted action-space BFS: a program-independent return upper bound."""
    queue = deque([(START[0], START[1], START[2], 0, 0)])
    seen = set()
    full_mask = (1 << len(CHECKPOINTS)) - 1
    while queue:
        row, col, direction, mask, steps = queue.popleft()
        if (row, col) == GOAL and mask == full_mask:
            return steps
        state = (row, col, direction, mask)
        if state in seen:
            continue
        seen.add(state)
        for action in range(3):
            nr, nc, nd = row, col, direction
            if action == 0 and clear(row, col, direction):
                dr, dc = DIRECTIONS[direction]
                nr, nc = row + dr, col + dc
            elif action == 1:
                nd = (direction - 1) % 4
            elif action == 2:
                nd = (direction + 1) % 4
            new_mask = mask
            for i, point in enumerate(CHECKPOINTS):
                if (nr, nc) == point:
                    new_mask |= 1 << i
            queue.append((nr, nc, nd, new_mask, steps + 1))
    raise ValueError("goal and checkpoints not mutually reachable")
