"""Report new-task tracking measurements, without directed-course assumptions."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    configuration = json.loads((a.run / "config.json").read_text())
    if configuration.get("task_spec", {}).get("task_version") != 2:
        raise ValueError("This report expects a new rough-task run")
    metrics = [json.loads(line) for line in (a.run / "metrics.jsonl").read_text().splitlines()]
    last = metrics[-1]
    lines = [
        "# 多方向混合地形訓練報告",
        "",
        f"Run：`{a.run.name}`。",
        f"環境數 {configuration['args']['num_envs']}；最後 iteration {last['iteration'] + 1}。",
        f"訓練平均 episode {last['episode_seconds']:.2f} 秒，平面速度 RMSE "
        f"{last['planar_velocity_rmse']:.3f} m/s，角速度 RMSE {last['yaw_velocity_rmse']:.3f} rad/s。",
        "",
        "評估使用估測速度；分數為固定 20 秒內的線／角速度追蹤平均，提前終止後計零。",
        "短訓練驗證代表流程可運作，學習能力仍需長訓練與正式評估。",
        "",
        "| Checkpoint | 擾動 | 條件數 | 平均追蹤分數 | 平均跌倒率 | 平均存活秒數 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    rows = []
    for name in ("final", "best"):
        path = a.output / (name + "_evaluation.json")
        if not path.exists():
            continue
        results = json.loads(path.read_text())["results"]
        rows.extend({"checkpoint": name, **r} for r in results)
        for randomized in (False, True):
            selected = [r for r in results if r["randomized"] == randomized]
            if selected:
                lines.append(
                    f"| {name} | {'randomized' if randomized else 'nominal'} | "
                    f"{len(selected)} | {np.mean([r['tracking_score'] for r in selected]):.3f} | "
                    f"{np.mean([r['fallen_rate'] for r in selected]):.1%} | "
                    f"{np.mean([r['mean_duration_s'] for r in selected]):.2f} |"
                )
    lines += [
        "",
        "## 各地形與動作",
        "",
        "| Checkpoint | 地形 | 難度 | 動作 | 擾動 | 平面 RMSE | 角速度 RMSE | 分數 | 跌倒率 |",
        "|---|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['checkpoint']} | {r['terrain']} | {r['difficulty']:g} | {r['motion']} | "
            f"{'randomized' if r['randomized'] else 'nominal'} | "
            f"{r['planar_velocity_rmse']:.3f} | {r['yaw_velocity_rmse']:.3f} | "
            f"{r['tracking_score']:.3f} | {r['fallen_rate']:.1%} |"
        )
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "report.md").write_text("\n".join(lines) + "\n")
    (a.output / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
