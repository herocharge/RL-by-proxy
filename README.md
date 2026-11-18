# RL by Proxy: fixed-maze baseline experiment

An executable decision-list program controls a partially observing 8×8 robot. We compare uniform random search, direct hill climbing, and reward-only autoregressive REINFORCE under equal completed-program evaluation budgets. The constructor sees only its previous program choices. Execution traces are recorded for analysis, not used in learning.

The original maze admitted an optimal one-rule wall follower. Its outputs remain in `results/prior_easy/` and `handoff_prior_easy.zip`. The current fixed maze was validated before comparison: a four-rule program reaches all checkpoints and the goal, no one- or two-rule program reaches the goal, and full DSL enumeration confirms the best achievable return is 1.558. The baseline algorithms were not changed for the rerun.

```sh
python -m unittest discover -s tests
python experiment.py --validate
python experiment.py --budget 25 --seeds 0 --output results/smoke
python experiment.py --budget 500 --seeds 0 1 2 3 4 --output results/full
python validate_full_space.py --output results/full_space_validation.json
python plot.py --results results/full --figures handoff/figures
python make_handoff.py --results results/full
```

The last command creates `handoff/` and `handoff.zip`. Python 3.10+ and dependencies in `requirements.txt` are required. All runs use the fixed maze, 100 environment steps per program, CPU execution, and independent seeds. The reference program validates solvability and is never inserted into an optimizer's history.
