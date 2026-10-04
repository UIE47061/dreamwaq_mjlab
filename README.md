# DreamWaQ–mjlab：Little White V3

獨立 DreamWaQ 實作，目標為小白 V3 不使用相機／地形感測器，盲爬直線樓梯與斜坡。部署僅使用 IMU、關節位置／速度、指令和動作歷史；CENet 估測機身線速度及 latent context。

## 安裝與訓練（Linux／NVIDIA GPU）

```bash
cd ~/U1_ws/dreamwaq_mjlab
uv sync --extra train --extra dev --extra render
uv run --extra train dreamwaq-train --task mixed --num-envs 512 --iterations 10000 --seed 42 --run-name first_round
```

Flat 為 `--task flat`。`--iterations` 表示本次追加的 iterations。每 1,000 iterations 以估測速度執行地形 probe，最佳 probe policy 存為 `best.pt`，最後存為 `final.pt`，每 50 iterations 更新 `latest.pt`。TensorBoard：

```bash
uv run --extra train tensorboard --logdir logs
```

續訓需指定完整 DreamWaQ checkpoint 路徑，並保持相同環境數、task、seed 與 beta：

```bash
uv run --extra train dreamwaq-train --num-envs 512 --iterations 1000 --resume logs/RUN/latest.pt
```

續訓恢復網路、optimizers、normalizers、iteration、curriculum 和 RNG；重建模擬並開始新 episodes，不保證逐步重現中斷前的物理軌跡。既有 mjlab Flat／BlindStairs checkpoint 的網路格式不同，不可直接 resume。

完整首輪流水線（訓練、最後／最佳 checkpoint 各條件 100 次評估、匯出、CPU 影片與報告）：

```bash
uv run --extra train --extra render python scripts/first_round.py
```

對已完成 run 只執行後處理：

```bash
uv run --extra train --extra render python scripts/first_round.py --run logs/RUN
```

## 評估、Play 與匯出

```bash
uv run --extra train dreamwaq-eval logs/RUN/final.pt --trials 100 --mode both --output artifacts/final_evaluation.json
uv run dreamwaq-play logs/RUN/best.pt --terrain stairs_up --height 0.15
uv run dreamwaq-play logs/RUN/best.pt --terrain slope_up --angle 25
uv run dreamwaq-export logs/RUN/best.pt --output artifacts/exported
uv run dreamwaq-sim2sim artifacts/exported/policy.pt --terrain stairs_up --height 0.15
```

`dreamwaq-play` 是 CPU MuJoCo standalone viewer，方便同時檢查部署行為；GPU 能力測試使用 `dreamwaq-eval`。`--headless` 可無視窗執行，`--video artifacts/test.mp4` 會錄影（需 render extra；Linux 無視窗錄影可加 `MUJOCO_GL=egl`）。地形另有 `flat`、`stairs_down`、`slope_down`。

## Mac sim2sim

將此獨立專案與匯出的 `policy.pt`／`metadata.json` 複製到 Mac。安裝部署依賴，不需 Linux 訓練 extra：

```bash
cd dreamwaq_mjlab
uv sync
uv run mjpython -m dreamwaq_mjlab.deployment.sim2sim artifacts/exported/policy.pt --terrain stairs_up --height 0.15
uv run mjpython -m dreamwaq_mjlab.deployment.sim2sim artifacts/exported/policy.pt --terrain slope_up --angle 25
```

`policy.pt` 已包含 normalization、CENet 和 actor。輸入是 `[batch, 6, 45]` 的原始歷史，舊到新排列；每次 reset 用第一筆觀測填滿歷史。輸出 12 維關節殘差。不要重複 normalization，也不要把訓練 checkpoint 交給 `torch.jit.load()`。

Mac native viewer 尚未在本機 Mac 驗證；Linux CPU headless sim2sim 已有測試。專案不包含實體機器人通訊與控制部署。

## 驗收定義

5 階樓梯，階深 35 cm；連續坡面 3 m。前進指令 0.4 m/s，20 秒內到達障礙後的出口、沒有失敗終止才算成功；有指令但連續 3 秒進度不足 10 cm 記為卡住。各條件 100 次，nominal 僅抽樣起始姿態，randomized 加入完整模型／感測／推擾隨機化。

目標為上下 15 cm 樓梯與 ±25° 斜坡各方向 nominal ≥90%、randomized ≥80%。成功率由評估數據決定，訓練完成不等於達標。

## 測試

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra dev pytest -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra train --extra dev pytest -q -m gpu
uv run --extra dev ruff check src tests scripts
```

停用外部 pytest plugins 可避免工作站 ROS plugins 干擾本專案。

模型／版本來源及實作差異見 [docs/design.md](docs/design.md)。Git 排除訓練成果與大型輸出，程式、模型、lockfile、測試和數值報告可獨立推送。

若首輪流水線中斷，可接續到總計 10,000 iterations，再自動評估與匯出：

```bash
uv run --extra train --extra render python scripts/first_round.py --resume logs/RUN/latest.pt
```
