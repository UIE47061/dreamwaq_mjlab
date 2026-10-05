# 實作與來源

## 模型和依賴

- 小白模型：[morrisx28/csl_mujoco](https://github.com/morrisx28/csl_mujoco/tree/0ab74ef5d6048345db7a541316019100416b57d6/unitree_robots/little_white_v3)，commit `0ab74ef5d6048345db7a541316019100416b57d6`。
- mjlab：[mujocolab/mjlab](https://github.com/mujocolab/mjlab/tree/b517e0c489139e7fcee95702cfb2b01931264985)，commit `b517e0c489139e7fcee95702cfb2b01931264985`。RSL-RL `5.5.0` 為鎖定的 mjlab 依賴。
- DreamWaQ：[論文](https://arxiv.org/html/2301.10602)。本專案依方法重新實作 CENet、β-VAE、AdaBoot 和 PPO，不移植 Isaac Gym simulator，也不修改安裝的 RSL-RL。

保留來源原始 `little_white_v3.xml`，訓練／部署共同使用適配後的 `robot.xml`。身體／關節／慣量／幾何參數不變；新增足部 sites、命名感測器，開啟完整碰撞並設定摩擦和接觸維度，碰撞 margin 設為 0 以符合 MuJoCo Warp MULTICCD。mjlab 載入時以同等 torque motors 取代 XML motors，按名稱映射。來源資產 BSD-3-Clause 授權隨資產保存；mjlab 適配程式依 Apache-2.0，授權在根目錄及 docs/LICENSE_MJLAB。

關節順序固定 FR、FL、RL、RR，每腿 hip、thigh、calf；動作目標 clamp 到來源硬限制。KP 20、KD 0.5、torque limit 27 Nm、action scale 0.25，physics 200 Hz／policy 50 Hz。

GPU 與 CPU 都使用 implicitfast、Newton 10 iterations、line search 20、CCD 50、elliptic cone 和 impratio 10，足部 contact priority 1。新版機身接地終止採最近四個 physics steps 中超過 1 N 的 trunk 接觸力；大腿接觸僅懲罰。CPU 每個控制步後執行 forward 更新姿態與感測器。

新版指令為機身 yaw frame 的平面速度與 yaw rate，隨機目標朝向透過 heading controller 轉換成角速度指令。Actor 只接收最終三維速度指令，不接收絕對朝向。CPU 平地 sim2sim 同樣直接使用機身指令；metadata 以 task_version=2 與 body_yaw_velocity 辨識，拒絕舊版世界座標指令模型。

## 網路與資料流

Proprioception 45 維：angular velocity 3、projected gravity 3、command 3、relative joint position 12、joint velocity 12、last action 12。CENet 使用目前和前 5 步的 6 幀歷史（270 維），ELU encoder 128→64，共享 encoder 輸出速度 3、latent 均值 16 及 log variance 16。Decoder 64→128→45 重建下一筆觀測。Loss 為 velocity MSE + next-observation MSE + β KL，預設 β=1。

PPO actor 接收 normalization 後目前觀測、3 維速度與 16 維 latent（64 維），網路 512→256→128→12。Critic 使用 259 維特權觀測，包括真實速度、raycast 和腳部接觸；同樣使用 512→256→128。Actor 的部署圖只接受 proprioception history。

訓練用採樣 latent；部署用 latent 均值。AdaBoot 初始採真實速度，之後使用 `1-tanh(std(episode_returns)/mean(episode_returns))` 決定採估測速度機率；不足兩個 episode 或均值非正時取 0。評估與部署全程使用估測速度。

每次 rollout 保存當時的 actor features、Gaussian params、log probabilities、critic features；更新 PPO 時固定 CENet，不重算 context，然後獨立更新 CENet。每步取得 reset 前的 terminal observation，正確處理 timeout bootstrap，重建 loss 遮罩所有 episode 結束 transition。History reset 僅影響結束的環境，無跨 episode 資訊。

Normalization 在每個 rollout 期間固定，rollout 結束才更新統計。Critic 有獨立 normalizer。PPO 設定：24 rollout steps、5 epochs、4 minibatches、clip .2、γ .99、GAE λ .95、entropy .01、gradient norm 1，adaptive learning rate 1e-5～1e-3。CENet Adam learning rate 1e-3，gradient norm 1。

## 地形、課程與原版差異

參考 [Quadruped_Dwaq_Lab / ROUGH_TERRAINS_CFG](https://github.com/UIE47061/Quadruped_Dwaq_Lab/blob/a8c43c943fc76349ef8e43f81cfd99f3ad90e66c/TienKung-Lab/legged_lab/terrains/terrain_generator_cfg.py)。此版本的 LittleWhiteDwaqEnvCfg 使用 ROUGH_TERRAINS_CFG，而非 DWAQ_TERRAINS_CFG。任務移植保留地形參數、比例、指令分布與距離課程；保留本專案網路、控制、一般獎勵及隨機化，並非完整複製舊版獎勵／演算法／物理引擎。

mjlab 原生 MuJoCo box 生成正／倒金字塔樓梯及隨機格子，heightfield 生成粗糙、波浪和坡面。每塊 8×8 m，10 級難度；每種地形一欄，比例決定機器人分配，與 Isaac Lab 20 欄配置不同。

階高參數 0～23 cm。mjlab 階數為 floor((8−2×1−3)/(2×階深))，分別 5、4、5、4；中央實際平台邊長約 3.20、3.44、3.00、3.28 m。Isaac Lab 2.1 另加一圈台階、內部平台更小，中心高度也不同；本版沿用 mjlab 原生幾何，不聲稱逐頂點一致。實際高度與出生點以幾何測試／預覽確認。

隨機格子的高度是在 ±bound 取樣，中心平臺高度為正 bound；rough 噪聲固定 −2～4 cm，不隨課程難度縮放。波浪與坡面沿用 mjlab 原生 heightfield；MuJoCo heightfield 接觸和 Isaac Lab 的三角網格接觸不同，斜坡陡邊不套用 Isaac 的 slope_threshold 網格校正。

雙層坑依 [Isaac Lab 2.1 pit_terrain](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains.py) 實作：外圈頂面 0、內圈 −d、中心 −2d，中圈邊長為 platform_width + (size−platform_width)×0.6。d=0～0.3 m；以實心底板避免零難度產生零厚度碰撞體。保留原設定名稱 high_platform，但它實際是坑。

初始難度 0～5；平面距出生原點 >4 m 升級，低於目前指令平面速度×20秒×0.5 降級，升級優先。首次 reset 不更新；resume 重建 episodes 也暫停一次課程更新。站立／旋轉沒有單向卡住判定。課程使用終點位移而非積分路程，與參考版本一致。

## 隨機化、評估與版本

隨機化沿用本專案摩擦各軸、質量／pseudo inertia、PD、motor strength、encoder bias、控制延遲、推擾及 proprioception 雜訊。Nominal 關閉模型／感測／推擾擾動，但仍抽樣起始 XY／yaw；actor 與 clean 分開，critic 無觀測雜訊。

平面追蹤使用實際世界線速度轉到機身 yaw frame，角速度追蹤使用世界 z yaw rate；避免傾斜時把完整 body frame 與 yaw frame 混用。速度估測監督目標仍為三維 body-frame 真實速度。

Probe 和正式評估使用 CENet 估測速度及 latent 均值。每個條件固定 20 秒，分數為線／角速度指數追蹤各半，提前終止後計零，條件等權平均。評估一次生成全部所需地形及難度，最多 128 個環境分批執行，保護 Python／NumPy／PyTorch／CUDA RNG；固定動作在 command resampling 後重新套用，random 條件保留 heading controller。

Checkpoint 存 task version、指令座標、完整地形／指令配置、task、seed、beta、環境數。Resume 必須匹配全部配置，舊單向 checkpoint 不能續訓／Play／匯出。既有資料保留供歷史參考。

訓練使用 explicit reset 前 terminal observation，time_out／地圖越界保留 value bootstrap，跌倒／機身接觸遮罩 bootstrap。Play 採 auto reset，依 episode_length_buf 分別重置六幀歷史。CPU sim2sim 只驗證平地部署；新版混合地形由 mjlab 執行。
