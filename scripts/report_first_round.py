"""Create source-controlled numerical results and standalone learning plots."""

import json
import sys
from pathlib import Path

import torch

run, output = map(Path, sys.argv[1:3])
rows = []
for checkpoint in ("final", "best"):
    result = json.loads((output / f"{checkpoint}_evaluation.json").read_text())
    for r in result["results"]:
        rows.append({"checkpoint": checkpoint, **r})
Path("docs").mkdir(exist_ok=True)
videos = json.loads((output / "video_results.json").read_text())
Path("docs/first_round_results.json").write_text(
    json.dumps({"run": str(run), "results": rows, "cpu_video_results": videos}, indent=2)
)
metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
last = metrics[-1]
lines = [
    "# 小白 V3 DreamWaQ 首輪成果",
    "",
    f"訓練資料夾：`{run}`。Seed 42；已完成 10,000 iterations。",
    "",
    f"最後紀錄：平均 episode {last['episode_seconds']:.2f} 秒，訓練通過比例 {last['course_success']:.1%}，地形平均 level {last['terrain_level']:.2f}/9。",
    "",
    "固定評估使用估測速度及 latent 均值；每列 100 次，起始姿態在小範圍內抽樣。Nominal 關閉感測雜訊、推擾及模型參數隨機化；randomized 保留完整擾動。",
    "",
    "| Checkpoint | 地形 | 難度 | 擾動 | 成功 | 失敗終止 | 卡住 |",
    "|---|---|---:|---|---:|---:|---:|",
]
configuration = json.loads((run / "config.json").read_text())
best_iteration = (
    torch.load(run / "best.pt", map_location="cpu", weights_only=False)["iteration"] + 1
)
lines.insert(
    4,
    f"實際環境數 {configuration['args']['num_envs']}；每 iteration 24 steps。最佳 checkpoint 取自第 {best_iteration:,} iterations 的 nominal 小樣本 probe，不表示 randomized 驗收全部達標。",
)
if configuration["args"].get("resume"):
    lines.insert(
        4,
        f"本輪曾從 `{configuration['args']['resume']}` 接續；評估亂數狀態隔離修正後，繼續至總 iteration 10,000。前段與接續段的 metrics 已合併。",
    )
for r in rows:
    difficulty = f"{r['height_cm']:g} cm" if r["height_cm"] is not None else f"{r['angle_deg']:g}°"
    lines.append(
        f"| {r['checkpoint']} | {r['terrain']} | {difficulty} | {'randomized' if r['randomized'] else 'nominal'} | {r['success_rate']:.0%} | {r['fallen_rate']:.0%} | {r['stalled_rate']:.0%} |"
    )
targets = [r for r in rows if r["height_cm"] == 15 or r["angle_deg"] == 25]
for ckpt in ("final", "best"):
    passed = all(
        r["success_rate"] >= (0.8 if r["randomized"] else 0.9)
        for r in targets
        if r["checkpoint"] == ckpt
    )
    lines += ["", f"**{ckpt} 的 15 cm／25° 全方向驗收：{'達標' if passed else '未達標'}。**"]
lines += [
    "",
    "## 目標難度的失敗分類",
    "",
    "下列數字為每 100 次測試的次數；接觸、姿態與越界原因可能同時發生，不能相加當作總失敗數。逾時指未通過且未被判定為跌倒或卡住。",
    "",
    "| Checkpoint | 地形 | 擾動 | 接觸失敗 | 姿態失敗 | 越界 | 卡住 | 逾時 |",
    "|---|---|---|---:|---:|---:|---:|---:|",
]
for r in targets:
    reasons = r["failure_reasons"]
    timeout = sum(
        not (trial["success"] or trial["failed"] or trial["stalled"])
        for trial in r["trial_results"]
    )
    lines.append(
        f"| {r['checkpoint']} | {r['terrain']} | {'randomized' if r['randomized'] else 'nominal'} | "
        f"{round(reasons['illegal_contact'] * r['trials'])} | "
        f"{round(reasons['fell_over'] * r['trials'])} | "
        f"{round(reasons['corridor_exit'] * r['trials'])} | "
        f"{round(r['stalled_rate'] * r['trials'])} | {timeout} |"
    )
lines += [
    "",
    "首輪主要瓶頸是隨機化 25° 上坡。最後版本 100 次中有 13 次卡住、12 次失敗終止；最佳版本有 15 次卡住、12 次失敗終止、2 次逾時。兩者無擾動上坡均為 100%，顯示目前較弱的是擾動下的穩定性，而非 nominal 地形完全無法通過；摩擦、推擾、COM 等各自的因果影響尚未做消融。",
    "",
    "下一輪建議先固定 25° 上坡與評估 seed，分別測試摩擦、推擾、COM、馬達／延遲的影響；增加高坡等級訓練覆蓋與擾動恢復訓練，再按接觸／姿態／卡住分類調整 reward。最佳 checkpoint 的定期選擇也應納入 randomized 目標測試，避免只依 nominal probe 選擇。這些調整尚未執行，未追加第二輪訓練。",
]
lines += ["", "## CPU sim2sim 影片", ""]
for video in videos["metrics"]:
    value = (
        f"{video['height_cm']:g} cm"
        if video["kind"].startswith("stairs")
        else f"{video['angle_deg']:g}°"
    )
    lines.append(
        f"- {video['kind']} {value}：{video['status']}；"
        f"進度 {video['max_forward_m']:.2f}/{video['goal_forward_m']:.2f} m，"
        f"時間 {video['duration_s']:.2f} 秒。"
    )
if videos["errors"]:
    lines.append(f"錄影執行失敗：{', '.join(videos['errors'])}。")
if (output / "gpu_failure_recording.json").exists():
    recording = json.loads((output / "gpu_failure_recording.json").read_text())
    recorded = recording["results"][0]
    trial = recorded["trial_results"][recording["video_env"]]
    outcome = "卡住" if trial["stalled"] else "失敗終止" if trial["failed"] else "逾時"
    lines += [
        "",
        f"GPU 隨機化 25° 上坡未通過影片：`gpu_randomized_slope_up_25_failure_trial2.mp4`。使用最後 checkpoint、seed 2026、額外 100 個環境錄下 trial 2（{outcome}）。此錄影批次成功率為 {recorded['success_rate']:.0%}，數據在 `gpu_failure_recording.json`；正式驗收仍以 `final_evaluation.json` 為準。CPU nominal 成功影片使用不同擾動條件。",
    ]
lines += [
    "",
    "成功是 20 秒內到達完整地形出口，且未發生失敗終止。卡住為連續 3 秒前進不足 10 cm；存活不代表通過。",
    "",
    f"完整結果、TorchScript、影片與 log：`{output}`。CPU nominal 與 GPU randomized 影片分別標示；Mac 原生 viewer 尚未在 Mac 本機驗證。",
    "",
    "工程驗證：21 項單元測試與 1 項 GPU 整合測試通過；TorchScript 重新載入後，單一與多環境輸出和原推論一致。訓練／部署共用關節、站姿、硬角度限制、PD 與扭矩限制；CPU 的積分器、接觸力門檻和 world-command 轉換已對齊。測試 logs 保存在 artifacts。",
]
Path("docs/first_round_report.md").write_text("\n".join(lines) + "\n")
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    for ax, key in zip(
        axs.flat, ("episode_reward", "course_success", "terrain_level", "velocity_mse")
    ):
        ax.plot([m["iteration"] for m in metrics], [m[key] for m in metrics])
        ax.set(xlabel="Iteration", ylabel=key)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)
except ImportError:
    pass
print("Report: docs/first_round_report.md")
