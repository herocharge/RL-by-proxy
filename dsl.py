"""Nine-component, four-rule decision-list programs."""
from __future__ import annotations

from dataclasses import dataclass
import random

PREDICATES = ("front_clear", "left_clear", "right_clear", "not_front_clear", "not_left_clear", "not_right_clear")
ACTIONS = ("move", "turn_left", "turn_right")
SIZES = (6, 3, 6, 3, 6, 3, 6, 3, 3)


@dataclass(frozen=True)
class Program:
    choices: tuple[int, ...]

    def __post_init__(self):
        if len(self.choices) != 9 or any(not 0 <= x < SIZES[i] for i, x in enumerate(self.choices)):
            raise ValueError("invalid program components")

    @property
    def id(self):
        return "-".join(map(str, self.choices))

    def text(self):
        return "\n".join(
            [f"IF {PREDICATES[self.choices[2*i]]} -> {ACTIONS[self.choices[2*i+1]]}" for i in range(4)]
            + [f"ELSE -> {ACTIONS[self.choices[8]]}"]
        )

    def act(self, obs: tuple[bool, bool, bool]) -> tuple[int, int]:
        for i in range(4):
            p = self.choices[2*i]
            if (obs[p % 3] if p < 3 else not obs[p % 3]):
                return self.choices[2*i+1], i
        return self.choices[8], 4


def random_program(rng: random.Random) -> Program:
    return Program(tuple(rng.randrange(n) for n in SIZES))


def mutate(program: Program, rng: random.Random) -> Program:
    i = rng.randrange(9)
    choices = list(program.choices)
    choices[i] = (choices[i] + rng.randrange(1, SIZES[i])) % SIZES[i]
    return Program(tuple(choices))
