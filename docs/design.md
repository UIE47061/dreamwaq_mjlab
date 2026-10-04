# 實作與來源

## 模型和依賴

- 小白模型：[morrisx28/csl_mujoco](https://github.com/morrisx28/csl_mujoco/tree/0ab74ef5d6048345db7a541316019100416b57d6/unitree_robots/little_white_v3)，commit `0ab74ef5d6048345db7a541316019100416b57d6`。
- mjlab：[mujocolab/mjlab](https://github.com/mujocolab/mjlab/tree/b517e0c489139e7fcee95702cfb2b01931264985)，commit `b517e0c489139e7fcee95702cfb2b01931264985`。RSL-RL `5.5.0` 為鎖定的 mjlab 依賴。
- DreamWaQ：[論文](https://arxiv.org/html/2301.10602)。本專案依方法重新實作 CENet、β-VAE、AdaBoot 和 PPO，不移植 Isaac Gym simulator，也不修改安裝的 RSL-RL。

保留來源原始 `little_white_v3.xml`，訓練／部署共同使用適配後的 `robot.xml`。身體／關節／慣量／幾何參數不變；新增足部 sites、命名感測器，開啟完整碰撞並設定摩擦和接觸維度，碰撞 margin 設為 0 以符合 MuJoCo Warp MULTICCD。mjlab 載入時以同等 torque motors 取代 XML motors，按名稱映射。來源資產 BSD-3-Clause 授權隨資產保存；mjlab 適配程式依 Apache-2.0，授權在根目錄及 docs/LICENSE_MJLAB。

關節順序固定 FR、FL、RL、RR，每腿 hip、thigh、calf；動作目標 clamp 到來源硬限制。KP 20、KD 0.5、torque limit 27 Nm、action scale 0.25，physics 200 Hz／policy 50 Hz。

GPU 與 CPU 都使用 implicitfast、Newton 10 iterations、line search 20、CCD 50、elliptic cone 和 impratio 10，足部 contact priority 1。大腿接觸終止採最近四個 physics steps 中超過 10 N 的接觸力，並非一接觸即失敗。CPU 每個控制步後執行 forward 更新姿態與感測器。

直線測試的速度指令固定在地形世界座標；輸入 actor 前，按當前 yaw 轉到機身座標，與 mjlab world command 的定義一致。CPU sim2sim 使用模擬 IMU 姿態的 yaw 進行這項指令轉換，沒有輸入真實線速度、地形高度或接觸。若後續接實體控制器，需明確選擇機身速度指令或另行維護 IMU heading 參考；本輪不交付實體控制器。

## 網路與資料流

Proprioception 45 維：angular velocity 3、projected gravity 3、command 3、relative joint position 12、joint velocity 12、last action 12。CENet 使用目前和前 5 步的 6 幀歷史（270 維），ELU encoder 128→64，共享 encoder 輸出速度 3、latent 均值 16 及 log variance 16。Decoder 64→128→45 重建下一筆觀測。Loss 為 velocity MSE + next-observation MSE + β KL，預設 β=1。

PPO actor 接收 normalization 後目前觀測、3 維速度與 16 維 latent（64 維），網路 512→256→128→12。Critic 使用 259 維特權觀測，包括真實速度、raycast 和腳部接觸；同樣使用 512→256→128。Actor 的部署圖只接受 proprioception history。

訓練用採樣 latent；部署用 latent 均值。AdaBoot 初始採真實速度，之後使用 `1-tanh(std(episode_returns)/mean(episode_returns))` 決定採估測速度機率；不足兩個 episode 或均值非正時取 0。評估與部署全程使用估測速度。

每次 rollout 保存當時的 actor features、Gaussian params、log probabilities、critic features；更新 PPO 時固定 CENet，不重算 context，然後獨立更新 CENet。每步取得 reset 前的 terminal observation，正確處理 timeout bootstrap，重建 loss 遮罩所有 episode 結束 transition。History reset 僅影響結束的環境，無跨 episode 資訊。

Normalization 在每個 rollout 期間固定，rollout 結束才更新統計。Critic 有獨立 normalizer。PPO 設定：24 rollout steps、5 epochs、4 minibatches、clip .2、γ .99、GAE λ .95、entropy .01、gradient norm 1，adaptive learning rate 1e-5～1e-3。CENet Adam learning rate 1e-3，gradient norm 1。

## 地形、randomization 和評估

使用有方向的 5 階直線樓梯與 3 m 連續斜坡。出生在入口平台，出口在障礙後方；同一幾何生成程式供 GPU 及 CPU 使用。訓練比例各 20%：平地、上／下樓梯、上／下坡；10 difficulty levels，階高 2～15 cm、階深 30～40 cm、坡度 0～25°。

課程在通過且未失敗時升級，其他結束／卡住時降級，首次 reset 不變動。訓練的樓梯出口距離取最長階深的保守出口；固定評估依 35 cm 階深計算確切出口。Reward 沿用現有小白 rough 的速度、姿態、動作平滑、足部與接觸適配，加向前進度 reward 0.5；卡住終止加 -0.2 penalty。

預設 randomization：

- 起點 x ±0.1 m、y ±0.15 m、yaw ±0.15 rad，額外 z 0.01～0.05 m；前進速度指令 0.2～0.6 m/s。
- Actor 觀測 uniform noise ±0.02（統一基準，與原論文各感測尺度設定不同）；critic／CENet supervision 無動態觀測雜訊。持續 encoder bias ±0.015 rad。
- 腳部 sliding friction 0.3～1.5、spin friction 1e-4～2e-2、rolling friction 1e-5～5e-3（本模型 condim 3，spin/rolling 參數已保存但不產生相應自由度摩擦力）。
- 機身密度 scale 0.9～1.1、COM x/y ±2.5 cm、z ±3 cm，使用 pseudo-inertia 同時調整質量／慣量，保持物理一致。
- KP、KD、motor strength 各 0.9～1.1；PD torque 先乘 strength，再 clamp ±27 Nm。
- Physics-step command delay 0～3 steps，即 0～15 ms，每 200 physics steps 重抽。
- 每 1～3 秒推擾：linear velocity x/y ±0.5、z ±0.4 m/s，angular roll/pitch ±0.52、yaw ±0.78 rad/s。

Nominal 評估保留小範圍起始姿態抽樣、關閉其餘擾動；randomized 使用以上全部設定。100 次包含不同起始姿態而非重複完全相同的 rollout。每次 20 s，前進速度 0.4 m/s；出口、跌倒／非法接觸／越界及卡住決定結果。

## 與原論文的差異與限制

這是小白 V3 和 MuJoCo Warp 的方法移植，並非 Unitree A1／Isaac Gym 實驗逐項重現。使用自己的 PPO runner、上述網路／history 解讀、定向地形、小白控制參數和 reward。模型與接觸參數變更都有測試，實體馬達的熱模型／完整 torque-speed curve 尚未建模。盲爬無法在尚未接觸前取得前方樓梯幾何；學習出來的能力需以評估驗證。

定期 probe 與主訓練的 Python／NumPy／PyTorch／CUDA RNG 狀態隔離。Probe 使用 15 cm／25° 的四方向 nominal 測試，每方向 5 次；最佳 checkpoint 主要按平均通過率選擇，同分時用前進距離作微小 tie-break。完整版能力驗收仍為每條件 100 次。
