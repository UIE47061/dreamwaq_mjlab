"""Train a new rough-task round, then evaluate, export and record it."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

from dreamwaq_mjlab.checkpoint import validate_checkpoint


def call(module, args, log):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        subprocess.run(
            [sys.executable, "-m", module, *map(str, args)],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--run", type=Path, help="Only postprocess an existing new-task run")
    source.add_argument("--resume", type=Path)
    p.add_argument("--iterations", type=int, default=10000, help="Additional iterations")
    p.add_argument("--num-envs", type=int)
    p.add_argument("--task", choices=("flat", "mixed"))
    p.add_argument("--seed", type=int)
    p.add_argument("--beta", type=float)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    Path("artifacts").mkdir(exist_ok=True)
    if a.run is None:
        saved = {}
        if a.resume:
            state = torch.load(a.resume, map_location="cpu", weights_only=False)
            validate_checkpoint(state)
            saved = state["config"]
        values = {
            key: getattr(a, key) if getattr(a, key) is not None else saved.get(key, default)
            for key, default in (("num_envs", 512), ("task", "mixed"), ("seed", 42), ("beta", 1.0))
        }
        args = [
            "--iterations",
            a.iterations,
            "--eval-interval",
            a.eval_interval,
            "--run-name",
            "rough_round",
        ]
        for key, value in values.items():
            args += ["--" + key.replace("_", "-"), value]
        if a.resume:
            args += ["--resume", a.resume]
        log = Path("artifacts/rough_round_train.log")
        call("dreamwaq_mjlab.learning.train", args, log)
        match = re.search(r"^RUN_DIR=(.+)$", log.read_text(), re.MULTILINE)
        if match is None:
            raise RuntimeError(f"Training did not report RUN_DIR; inspect {log}")
        run = Path(match.group(1))
    else:
        run = a.run
    output = a.output or Path("artifacts") / (run.name + "_evaluation")
    output.mkdir(parents=True, exist_ok=True)
    # EGL is only required by the subprocess that records the video.
    os.environ.setdefault("MUJOCO_GL", "egl")
    for name in ("final", "best"):
        checkpoint = run / (name + ".pt")
        if not checkpoint.exists():
            continue
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        validate_checkpoint(state)
        terrain = "flat" if state["task_spec"]["task"] == "flat" else "mixed"
        call(
            "dreamwaq_mjlab.deployment.evaluate",
            [
                checkpoint,
                "--terrain",
                terrain,
                "--trials",
                a.trials,
                "--output",
                output / (name + "_evaluation.json"),
            ],
            output / (name + "_evaluation.log"),
        )
        call(
            "dreamwaq_mjlab.deployment.export",
            [checkpoint, "--output", output / name],
            output / (name + "_export.log"),
        )
        call(
            "dreamwaq_mjlab.deployment.play",
            [
                checkpoint,
                "--terrain",
                terrain,
                "--seconds",
                20,
                "--headless",
                "--video",
                output / (name + "_mixed.mp4"),
            ],
            output / (name + "_play.log"),
        )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("report_training.py")),
            str(run),
            str(output),
        ],
        check=True,
    )
    (output / "run.json").write_text(json.dumps({"run": str(run), "output": str(output)}, indent=2))
    print(output.resolve())


if __name__ == "__main__":
    main()
