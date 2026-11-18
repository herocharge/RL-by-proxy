"""Generate learning curves and execution-path figures from saved results."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dsl import Program
from environment import MAZE, START, CHECKPOINTS, GOAL, evaluate


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def trace_plot(program, title, output):
    result = evaluate(program, trace=True)
    fig = plt.figure(figsize=(12, 7), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1, 1])
    ax = fig.add_subplot(grid[:, 0])
    for row in range(8):
        for col in range(8):
            ax.add_patch(plt.Rectangle((col, row), 1, 1, facecolor="0.2" if MAZE[row][col] == "#" else "white", edgecolor="0.75"))
    path = [START[:2]] + [(r["next_environment_state"]["row"], r["next_environment_state"]["col"]) for r in result.trace]
    ax.plot([p[1]+.5 for p in path], [p[0]+.5 for p in path], color="tab:blue", lw=2, alpha=.75)
    ax.scatter([START[1]+.5], [START[0]+.5], color="tab:green", s=130, marker="s", label="start")
    ax.scatter([GOAL[1]+.5], [GOAL[0]+.5], color="tab:red", s=150, marker="*", label="goal")
    for i, (r,c) in enumerate(CHECKPOINTS):
        ax.scatter([c+.5], [r+.5], color="tab:orange", s=90)
        ax.text(c+.5, r+.5, str(i+1), ha="center", va="center", fontsize=8)
    ax.set(xlim=(0,8), ylim=(8,0), aspect="equal", title="Maze and executed path", xticks=[], yticks=[])
    ax.legend(loc="lower left")
    ax2 = fig.add_subplot(grid[0,1])
    rewards = np.array([x["immediate_reward"] for x in result.trace])
    ax2.plot(np.arange(1,len(rewards)+1), np.cumsum(rewards), label="cumulative reward")
    events = [i+1 for i,x in enumerate(result.trace) if x["checkpoint_reached"] is not None]
    for event in events: ax2.axvline(event, color="tab:orange", alpha=.5)
    ax2.set(xlabel="Environment step", ylabel="Cumulative reward", title="Reward timeline")
    ax3 = fig.add_subplot(grid[1,1])
    ax3.axis("off")
    info = f"{program.text()}\n\nRule activations: {result.activation_counts}\nReward: {result.reward:.3f}; goal: {result.success}; steps: {result.steps}\nCheckpoint events: {events}"
    ax3.text(0, 1, info, va="top", family="monospace", fontsize=10)
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--figures", type=Path, default=Path("handoff/figures"))
    args=p.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    rows=read_csv(args.results/"all_evaluations.csv")
    fig,ax=plt.subplots(figsize=(9,5), layout="constrained")
    colors={"random":"tab:blue","hill":"tab:orange","reinforce":"tab:green"}
    for method in colors:
        groups={}
        for r in rows:
            if r["method"]==method:groups.setdefault(int(r["seed"]),[]).append(float(r["best_so_far"]))
        values=np.array([groups[k] for k in sorted(groups)])
        x=np.arange(1,values.shape[1]+1)
        mean=values.mean(axis=0)
        std=values.std(axis=0,ddof=1) if len(values)>1 else np.zeros_like(mean)
        ax.plot(x,mean,label=method,color=colors[method],lw=2)
        ax.fill_between(x,mean-std,mean+std,color=colors[method],alpha=.15)
    ax.set(xlabel="Completed-program evaluations",ylabel="Best-so-far return",title="Fixed-maze program optimization")
    ax.legend()
    fig.savefig(args.figures/"baseline_learning_curves.png",dpi=200)
    plt.close(fig)
    successes=[r for r in rows if r["success"]=="True"]
    best=max(successes if successes else rows,key=lambda r:float(r["reward"]))
    failure_candidates=[r for r in rows if r["success"]=="False"]
    reinforce_failures=[r for r in failure_candidates if r["method"]=="reinforce"]
    failure=max(reinforce_failures or failure_candidates,key=lambda r:float(r["reward"])) if failure_candidates else min(rows,key=lambda r:float(r["reward"]))
    for label,row in ((("success" if successes else "best_observed"),best),("failure",failure)):
        program=Program(tuple(map(int,row["choices"].split("-"))))
        trace_plot(program,f"{label.capitalize()} example: {row['method']} seed {row['seed']} evaluation {row['evaluation']}",args.figures/f"execution_trace_{label}.png")
        result=evaluate(program,trace=True)
        (args.figures/f"execution_trace_{label}.json").write_text(json.dumps({"program":program.id,"reward":result.reward,"success":result.success,"trace":result.trace},indent=2))


if __name__=="__main__":main()
