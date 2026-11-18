import random
import unittest
from baselines import run
from dsl import Program, mutate, random_program
from environment import CHECKPOINTS, evaluate
from experiment import REFERENCE, validate


class CoreTests(unittest.TestCase):
    def test_decision_list_order_and_negation(self):
        p = Program((3, 2, 0, 0, 1, 1, 2, 0, 1))
        self.assertEqual(p.act((False, True, False)), (2, 0))
        self.assertEqual(p.act((True, True, False)), (0, 1))

    def test_determinism_and_reward_accounting(self):
        p = random_program(random.Random(4))
        a, b = evaluate(p, True), evaluate(p, True)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a.reward, sum(t["immediate_reward"] for t in a.trace))
        self.assertLessEqual(sum(t["checkpoint_reached"] is not None for t in a.trace), len(CHECKPOINTS))

    def test_mutation_changes_one_component(self):
        p = random_program(random.Random(3))
        q = mutate(p, random.Random(7))
        self.assertEqual(sum(x != y for x, y in zip(p.choices, q.choices)), 1)

    def test_budget_accounting(self):
        for method in ("random", "hill", "reinforce"):
            rows, summary = run(method, 0, 3)
            self.assertEqual(len(rows), 3)
            self.assertEqual(summary["budget"], 3)
            self.assertEqual([x["evaluation"] for x in rows], [1, 2, 3])

    def test_environment_difficulty(self):
        report = validate()
        self.assertTrue(report["reference_success"])
        self.assertEqual(report["reference_checkpoints"], 3)
        self.assertEqual(report["small_programs"]["1"]["successful"], 0)
        self.assertEqual(report["small_programs"]["2"]["successful"], 0)


if __name__ == "__main__":
    unittest.main()
