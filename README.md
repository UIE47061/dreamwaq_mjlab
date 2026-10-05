# DreamWaQ mjlab — Little White V3

使用 mjlab 訓練小白 V3 四足機器人，支援平地與多方向混合地形。
Actor 使用 IMU、關節狀態、速度指令及六幀歷史；CENet 估測速度與地形 latent，
critic 使用特權觀測。方法與來源詳見 [實作說明](docs/design.md)。

目前為 **task version 2 / reward revision `anti_bounce_v1`**。

## 1. 安裝

Train、Play、Eval 使用 Linux 與 NVIDIA CUDA GPU；獨立 sim2sim 使用 CPU MuJoCo。
Python 版本需為 3.11～3.13，依賴版本由 `uv.lock` 鎖定。

```bash
cd /home/uie/U1_ws/dreamwaq_mjlab
uv sync --extra train --extra dev --extra render
```

下列指令皆在這個資料夾執行。其他電腦請換成自己的專案路徑。

## 2. Train：訓練

混合地形，從頭訓練：

```bash
uv run --extra train dreamwaq-train \
  --task mixed \
  --num-envs 512 \
  --iterations 10000 \
  --seed 42 \
  --run-name anti_bounce_v1
```

只訓練平地時改成 `--task flat`。可用 `--num-envs 256` 降低記憶體需求；
預設 512 環境初始化遇到記憶體不足時，程式會嘗試改成 256。
`--iterations` 是本次追加的迭代數，每次 iteration 收集 24 個控制步。

開始後會印出 `RUN_DIR=...`，例如 `logs/日期時間_anti_bounce_v1`。
請將實際路徑填入以下變數；後續指令在同一個 shell 使用：

```bash
# 把 日期時間 換成 RUN_DIR 顯示的資料夾名稱。
RUN_DIR="logs/日期時間_anti_bounce_v1"
```

每個 run 的輸出：

| 檔案 | 用途 |
|---|---|
| `latest.pt` | 每 50 iterations 更新，供續訓或觀看 |
| `final.pt` | 正常完成訓練後儲存的最後 checkpoint |
| `best.pt` | 固定 probe 追蹤分數最佳的 checkpoint |
| `model_N.pt` | 第 N iterations probe 時儲存的 checkpoint |
| `config.json` | 訓練參數、任務與環境設定 |
| `metrics.jsonl` | Loss、速度追蹤誤差、跌倒比例等紀錄 |
| `probe_N.json` | 固定地形與動作 probe 結果 |
| `events.out.tfevents.*` | TensorBoard 紀錄 |

預設每 1,000 iterations 執行 probe。短於此間隔的訓練通常只有 latest／final，
沒有 best；未到 50 iterations 的訓練不會產生 latest。
**Train 不會自動匯出 sim2sim 所需的 `policy.pt`**，請執行第 4 節的 Export。

觀看訓練曲線：

```bash
uv run --extra train tensorboard --logdir logs
```

續訓目前 reward 的 run：

```bash
uv run --extra train dreamwaq-train \
  --task mixed --num-envs 512 --seed 42 --beta 1 \
  --iterations 2000 --run-name anti_bounce_resume \
  --resume "$RUN_DIR/latest.pt"
```

續訓要求 task、地形、指令、reward revision、實際環境數、seed、beta 相同。
若原 run 使用 256 環境，續訓也要填 256；新增紀錄會寫入新的 run 資料夾。
本次 reward 改版前的 checkpoint 無法直接 resume，請從頭訓練。
Task version 2 的舊 reward checkpoint 仍可 Play／Export；task version 1 不相容。

## 3. Play：觀看訓練結果

先用平地檢查是否仍有同步跳躍：

```bash
uv run --extra train dreamwaq-play "$RUN_DIR/final.pt" \
  --terrain flat --num-envs 8 --seconds 60
```

觀看混合地形：

```bash
uv run --extra train dreamwaq-play "$RUN_DIR/final.pt" \
  --terrain mixed --num-envs 16 --seconds 60
```

指定地形與難度：

```bash
uv run --extra train dreamwaq-play "$RUN_DIR/final.pt" \
  --terrain stairs_up_28 --difficulty 0.5 --num-envs 8
```

可換成 `best.pt`，或在訓練途中用 `latest.pt`。
Play 預設使用隨機速度／朝向指令、關閉模型與感測隨機化，出生姿態仍隨機。
`--difficulty` 範圍為 0～1；省略時隨機分配地形難度。
`--seconds` 是模擬時間。

有桌面時預設開 native viewer；無 DISPLAY 時預設開 Viser。
也可以明確指定瀏覽器 viewer，啟動後使用終端機印出的網址：

```bash
uv run --extra train dreamwaq-play "$RUN_DIR/final.pt" --viewer viser --seconds 60
```

無視窗錄影（Linux EGL）：

```bash
MUJOCO_GL=egl uv run --extra train --extra render dreamwaq-play \
  "$RUN_DIR/final.pt" --terrain mixed --headless --seconds 20 \
  --video "artifacts/$(basename "$RUN_DIR")/mixed.mp4"
```

## 4. Export PT：匯出 sim2sim 模型

訓練 checkpoint 與部署用 TorchScript 都是 `.pt`，但格式不同：

| 格式 | 產生方式 | 使用程式 |
|---|---|---|
| `final.pt`／`best.pt`／`latest.pt` | Train | Play、Eval、Resume、Export |
| `policy.pt` | Export | sim2sim、`torch.jit.load()` |

建立本次 run 的獨立輸出路徑並匯出：

```bash
EXPORT_DIR="artifacts/$(basename "$RUN_DIR")/exported"

uv run --extra train dreamwaq-export "$RUN_DIR/final.pt" \
  --output "$EXPORT_DIR"
```

輸出為 `policy.pt` 和 `metadata.json`。Export 會重新載入 TorchScript，
檢查單一及批次輸出與原模型一致。也可將來源換成 `best.pt`。
重用同一個 output 路徑會覆寫匯出檔案。

`policy.pt` 已包含 normalization、CENet 和 actor：
輸入為原始 proprioception history `[batch, 6, 45]`（舊到新），輸出為 12 維關節殘差。
Reset 時以第一幀填滿歷史，不要再次 normalization。
關節、PD、動作比例與指令座標資訊記錄在 `metadata.json`。

## 5. sim2sim：CPU MuJoCo 平地測試

先完成 Export，再將 `policy.pt` 傳入：

```bash
uv run dreamwaq-sim2sim "$EXPORT_DIR/policy.pt" \
  --command 0.4 0 0 --seconds 60
```

`--command` 依序是機身 yaw frame 的 vx（m/s）、vy（m/s）、yaw rate（rad/s）。
例如後退、側移加轉向：

```bash
uv run dreamwaq-sim2sim "$EXPORT_DIR/policy.pt" \
  --command -0.4 0.2 0.8 --seconds 20
```

無視窗測試：

```bash
uv run dreamwaq-sim2sim "$EXPORT_DIR/policy.pt" \
  --command 0.4 0 0 --headless --seconds 20
```

此程式僅用於 CPU 平地部署檢查；混合地形驗證使用 Play／Eval。
不要直接把 `final.pt` 傳給 sim2sim。

## 6. Eval：評估

先做小樣本平地評估：

```bash
uv run --extra train dreamwaq-eval "$RUN_DIR/final.pt" \
  --terrain flat --difficulty 0 --motion forward standing \
  --trials 5 --mode nominal \
  --output "artifacts/$(basename "$RUN_DIR")/flat_evaluation.json"
```

完整評估涵蓋九種地形、三種難度、八種動作及 nominal／randomized：

```bash
uv run --extra train dreamwaq-eval "$RUN_DIR/final.pt" \
  --trials 100 --mode both \
  --output "artifacts/$(basename "$RUN_DIR")/evaluation.json"
```

完整評估耗時較長。結果包含追蹤分數、速度 RMSE、存活時間、跌倒與越界比例。
最佳 probe 分數不保證步態自然，仍需觀看影片確認跳躍與腳部協調。

## 7. Reward 與地形

`anti_bounce_v1` 保留原本速度追蹤、姿態、抬腳、滑動與碰撞 reward，新增／調整：

| 項目 | 計算方式 | 權重 |
|---|---|---:|
| `vertical_bounce` | `max(abs(world_vz) - 0.2, 0)²` | −1.0 |
| `prolonged_flight` | 四腳同時離地且持續超過 80 ms 時為 1 | −1.0 |
| `body_ang_vel` | 世界座標 roll／pitch 角速度平方和 | −0.05 |

短暫離地仍允許，沒有強制 trot。`air_time` reward 保持 0。
這些是抑制彈跳的起始權重，尚未完成新版 reward 的長訓練步態驗證。
世界座標垂直速度懲罰也可能影響快速爬坡／上下階梯，需和地形能力一起評估。
修改位置：`src/dreamwaq_mjlab/tasks/config.py`；計算函式在 `tasks/mdp.py`。

可選地形：`flat`、`mixed`，以及以下九種：

| 名稱 | 地形 |
|---|---|
| `stairs_up_28`、`stairs_up_32` | 正金字塔樓梯，階深 28／32 cm |
| `stairs_down_30`、`stairs_down_34` | 倒金字塔樓梯，階深 30／34 cm |
| `boxes` | 隨機高度方塊 |
| `random_rough` | 粗糙地面 |
| `wave` | 波浪地面 |
| `slope` | 金字塔坡面 |
| `high_platform` | 雙層坑，名稱沿用歷史設定 |

地形每塊 8 × 8 m，十級難度；樓梯最大階高 23 cm。
從中心往外走，正金字塔主要下樓、倒金字塔主要上樓。
速度指令涵蓋前後、左右及朝向控制，20% 環境抽樣站立。
Physics 200 Hz、policy 50 Hz；PD：KP 20、KD 0.5、扭矩上限 27 Nm、action scale 0.25。

## 8. 資料夾與輔助工具

```text
dreamwaq_mjlab/
├── README.md
├── pyproject.toml / uv.lock     # 安裝與依賴鎖定
├── LICENSE
├── src/dreamwaq_mjlab/
│   ├── tasks/                  # Reward、觀測、機器人與地形設定
│   ├── learning/               # CENet、PPO、訓練
│   ├── deployment/             # Play、Eval、Export、sim2sim
│   ├── assets/little_white_v3/  # XML、URDF、mesh、來源授權
│   ├── control.py              # 共用控制與關節順序
│   └── checkpoint.py           # 任務相容性檢查
├── scripts/
│   ├── preview_terrains.py      # 地形預覽
│   ├── train_pipeline.py       # 訓練／評估／匯出／錄影流水線
│   └── report_training.py      # 產生訓練與評估報告
├── tests/
├── docs/                       # 實作、歷史驗證與 mjlab 授權
├── logs/                       # 訓練 checkpoint 與紀錄
└── artifacts/                  # 匯出模型、影片、評估與歷史報告
```

地形預覽：

```bash
uv run --extra train --extra render python scripts/preview_terrains.py
```

已有 run 的自動評估、匯出、錄影與報告（先用小樣本）：

```bash
uv run --extra train --extra render python scripts/train_pipeline.py \
  --run "$RUN_DIR" --trials 5
```

從頭執行完整流水線：

```bash
uv run --extra train --extra render python scripts/train_pipeline.py \
  --task mixed --num-envs 512 --iterations 10000 --trials 5
```

流水線輸出預設在 `artifacts/<run名稱>_evaluation/`，
匯出的檔案分別為 `final/policy.pt` 與 `best/policy.pt`（若存在 best checkpoint）。
這是流水線腳本的功能，直接使用 `dreamwaq-train` 不會自動 Export。

測試與檢查：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra train --extra dev pytest -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra train --extra dev pytest -q -m gpu
uv run --extra dev ruff check src tests scripts
```

[歷史流程驗證](docs/rough_task_validation.md) 記錄 reward 改版前的測試。
Task version 1 的首輪報告已移到 `artifacts/history/task_v1/`。
`logs/` 與 `artifacts/` 為本機成果，不納入 Git；整理時請保留仍需使用的模型與紀錄。
