"""Run one authorized 10,000-iteration training round, then evaluate and export."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def call(module, args, log):
    with Path(log).open("w") as stream:
        subprocess.run(
            [sys.executable, "-m", module, *map(str, args)],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, help="Evaluate an already completed run")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    Path("artifacts").mkdir(exist_ok=True)
    if a.run is None:
        call(
            "dreamwaq_mjlab.learning.train",
            [
                "--num-envs",
                512,
                "--iterations",
                10000,
                "--seed",
                42,
                "--run-name",
                "first_round",
                "--eval-interval",
                1000,
            ],
            "artifacts/first_round_train.log",
        )
        runs = sorted(Path("logs").glob("*_first_round"))
        run = runs[-1]
    else:
        run = a.run
    (Path("artifacts") / "first_round_run.json").write_text(
        json.dumps({"run": str(run), "status": "evaluating"})
    )
    output = Path("artifacts") / run.name
    output.mkdir(exist_ok=True)
    for name in ("final", "best"):
        checkpoint = run / f"{name}.pt"
        call(
            "dreamwaq_mjlab.deployment.evaluate",
            [
                checkpoint,
                "--trials",
                100,
                "--mode",
                "both",
                "--output",
                output / f"{name}_evaluation.json",
            ],
            output / f"{name}_evaluation.log",
        )
        call(
            "dreamwaq_mjlab.deployment.export",
            [checkpoint, "--output", output / name],
            output / f"{name}_export.log",
        )
    # Videos are evidence of CPU sim2sim behavior and are labeled as such.
    video_errors = []
    video_metrics = []
    for kind in ("stairs_up", "stairs_down", "slope_up", "slope_down"):
        for value in (0.02, 0.15) if kind.startswith("stairs") else (10.0, 25.0):
            args = [
                output / "best/policy.pt",
                "--headless",
                "--seconds",
                20,
                "--terrain",
                kind,
                "--height",
                value if kind.startswith("stairs") else 0.15,
                "--angle",
                value if kind.startswith("slope") else 25.0,
                "--video",
                output / f"{kind}_{value}.mp4",
            ]
            logfile = output / f"video_{kind}_{value}.log"
            env = os.environ.copy()
            env["MUJOCO_GL"] = "egl"
            with logfile.open("w") as stream:
                done = subprocess.run(
                    [sys.executable, "-m", "dreamwaq_mjlab.deployment.sim2sim", *map(str, args)],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            if done.returncode:
                video_errors.append(str(logfile))
            else:
                for line in logfile.read_text().splitlines():
                    if line.startswith("{"):
                        video_metrics.append(json.loads(line))
    (output / "video_results.json").write_text(
        json.dumps({"metrics": video_metrics, "errors": video_errors}, indent=2)
    )
    subprocess.run(
        [sys.executable, "scripts/report_first_round.py", str(run), str(output)], check=True
    )
    (Path("artifacts") / "first_round_run.json").write_text(
        json.dumps({"run": str(run), "output": str(output), "status": "complete"})
    )
    print(f"Completed first round: {output}", flush=True)


if __name__ == "__main__":
    main()
