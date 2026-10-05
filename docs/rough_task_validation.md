# 多方向混合地形任務驗證

驗證日期：2026-10-04（Asia/Taipei）。環境：RTX 3060 Ti，8 GB VRAM。

這是 `anti_bounce_v1` reward 調整前的歷史驗證，不代表新版 reward 的步態成果。

新版任務、Train、Play、Eval、Export、CPU 平地 sim2sim 已實作；保留既有 logs 和成果，拒絕舊單向 checkpoint。

## 執行結果

- CPU：59 項通過；GPU：2 項通過；Ruff、格式與 uv lock 檢查通過。
- GPU 短訓練：64 環境、50 iterations；全部紀錄數值有限，未遇到 OOM。
- 第 25／50 iterations 各執行 216 條件 × 5 次 probe，共 2,160 次試驗。
- best.pt 取自第 25 iterations，固定 probe 平均分數 0.6610。
- 另追加 1 iteration 驗證 resume：任務配置、課程 levels/types、optimizer 狀態恢復，權重繼續更新。
- flat 任務另以 8 環境、1 iteration 驗證可訓練。
- 後處理流水線（現名 `scripts/train_pipeline.py`）通過：final／best 各 432 條件 × 2 次（nominal/randomized），合計 1,728 次。
- final／best 均完成匯出一致性測試及 20 秒、640×480、25 fps 混合地形錄影。
- Native 與 Viser 啟動／有限步數關閉通過；Native 已處理 MuJoCo 非同步關窗與 GLFW 結束的競爭。
- CPU 平地 sim2sim：後退＋側移＋旋轉指令 [-0.4, 0.2, 0.8]，2 秒完成，數值有限且 torque ≤27 Nm。

## 小樣本評估

| Checkpoint | 擾動 | 平均追蹤分數 | 跌倒率 | 平均存活秒數 |
|---|---|---:|---:|---:|
| final | nominal | 0.5796 | 3.0% | 19.45 |
| final | randomized | 0.5650 | 7.9% | 19.14 |
| best | nominal | 0.6537 | 3.5% | 19.37 |
| best | randomized | 0.6393 | 4.9% | 19.22 |

每條件僅 2 次，這些結果用於流程驗證。短訓練尚不足以宣稱能穩定前後、側向行走或通過所有地形；分數也會受到站立、低速指令與出生平台影響。正式能力驗收使用每條件 100 次，並檢視各動作／地形的速度 RMSE。

## 成果位置

- 訓練 run：`logs/2026-10-04_22-26-37_rough_smoke_verified`。
- 影片、完整各條件報告、匯出模型：`artifacts/rough_verified_validation`。
- 地形預覽：`artifacts/rough_terrain_preview.png`。
- 測試紀錄：`artifacts/rough_cpu_tests.log`、`artifacts/rough_gpu_tests.log`。
- 參考地形、階數與中心平台差異見 [design.md](design.md)。

## 可直接執行的指令

```bash
cd ~/U1_ws/dreamwaq_mjlab
uv run --extra train dreamwaq-play logs/2026-10-04_22-26-37_rough_smoke_verified/best.pt
uv run --extra train dreamwaq-train --task mixed --num-envs 512 --iterations 10000 --seed 42 --run-name rough_round
uv run --extra train dreamwaq-eval logs/RUN/best.pt --trials 100 --mode both --output artifacts/evaluation.json
```
