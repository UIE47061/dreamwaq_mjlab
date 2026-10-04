# 小白 V3 DreamWaQ 首輪成果

訓練資料夾：`logs/2026-10-04_19-22-18_first_round`。Seed 42；已完成 10,000 iterations。

本輪曾從 `logs/2026-10-04_19-01-21_first_round/latest.pt` 接續；評估亂數狀態隔離修正後，繼續至總 iteration 10,000。前段與接續段的 metrics 已合併。
實際環境數 512；每 iteration 24 steps。最佳 checkpoint 取自第 8,000 iterations 的 nominal 小樣本 probe，不表示 randomized 驗收全部達標。
最後紀錄：平均 episode 13.53 秒，訓練通過比例 91.6%，地形平均 level 5.97/9。

固定評估使用估測速度及 latent 均值；每列 100 次，起始姿態在小範圍內抽樣。Nominal 關閉感測雜訊、推擾及模型參數隨機化；randomized 保留完整擾動。

| Checkpoint | 地形 | 難度 | 擾動 | 成功 | 失敗終止 | 卡住 |
|---|---|---:|---|---:|---:|---:|
| final | stairs_up | 2 cm | nominal | 99% | 0% | 1% |
| final | stairs_up | 6 cm | nominal | 97% | 0% | 3% |
| final | stairs_up | 10 cm | nominal | 94% | 0% | 6% |
| final | stairs_up | 15 cm | nominal | 93% | 0% | 7% |
| final | stairs_down | 2 cm | nominal | 100% | 0% | 0% |
| final | stairs_down | 6 cm | nominal | 100% | 0% | 0% |
| final | stairs_down | 10 cm | nominal | 95% | 0% | 5% |
| final | stairs_down | 15 cm | nominal | 98% | 0% | 2% |
| final | slope_up | 10° | nominal | 100% | 0% | 0% |
| final | slope_up | 20° | nominal | 100% | 0% | 0% |
| final | slope_up | 25° | nominal | 100% | 0% | 0% |
| final | slope_down | 10° | nominal | 100% | 0% | 0% |
| final | slope_down | 20° | nominal | 100% | 0% | 0% |
| final | slope_down | 25° | nominal | 100% | 0% | 0% |
| final | stairs_up | 2 cm | randomized | 98% | 0% | 2% |
| final | stairs_up | 6 cm | randomized | 99% | 1% | 0% |
| final | stairs_up | 10 cm | randomized | 99% | 1% | 0% |
| final | stairs_up | 15 cm | randomized | 95% | 2% | 3% |
| final | stairs_down | 2 cm | randomized | 100% | 0% | 0% |
| final | stairs_down | 6 cm | randomized | 98% | 2% | 0% |
| final | stairs_down | 10 cm | randomized | 98% | 1% | 1% |
| final | stairs_down | 15 cm | randomized | 91% | 9% | 0% |
| final | slope_up | 10° | randomized | 100% | 0% | 0% |
| final | slope_up | 20° | randomized | 88% | 1% | 11% |
| final | slope_up | 25° | randomized | 75% | 12% | 13% |
| final | slope_down | 10° | randomized | 100% | 0% | 0% |
| final | slope_down | 20° | randomized | 97% | 1% | 0% |
| final | slope_down | 25° | randomized | 93% | 0% | 0% |
| best | stairs_up | 2 cm | nominal | 100% | 0% | 0% |
| best | stairs_up | 6 cm | nominal | 100% | 0% | 0% |
| best | stairs_up | 10 cm | nominal | 100% | 0% | 0% |
| best | stairs_up | 15 cm | nominal | 99% | 0% | 1% |
| best | stairs_down | 2 cm | nominal | 100% | 0% | 0% |
| best | stairs_down | 6 cm | nominal | 100% | 0% | 0% |
| best | stairs_down | 10 cm | nominal | 100% | 0% | 0% |
| best | stairs_down | 15 cm | nominal | 100% | 0% | 0% |
| best | slope_up | 10° | nominal | 100% | 0% | 0% |
| best | slope_up | 20° | nominal | 100% | 0% | 0% |
| best | slope_up | 25° | nominal | 100% | 0% | 0% |
| best | slope_down | 10° | nominal | 100% | 0% | 0% |
| best | slope_down | 20° | nominal | 100% | 0% | 0% |
| best | slope_down | 25° | nominal | 100% | 0% | 0% |
| best | stairs_up | 2 cm | randomized | 97% | 2% | 1% |
| best | stairs_up | 6 cm | randomized | 96% | 3% | 1% |
| best | stairs_up | 10 cm | randomized | 99% | 1% | 0% |
| best | stairs_up | 15 cm | randomized | 95% | 5% | 0% |
| best | stairs_down | 2 cm | randomized | 100% | 0% | 0% |
| best | stairs_down | 6 cm | randomized | 100% | 0% | 0% |
| best | stairs_down | 10 cm | randomized | 99% | 1% | 0% |
| best | stairs_down | 15 cm | randomized | 93% | 7% | 0% |
| best | slope_up | 10° | randomized | 100% | 0% | 0% |
| best | slope_up | 20° | randomized | 86% | 2% | 12% |
| best | slope_up | 25° | randomized | 71% | 12% | 15% |
| best | slope_down | 10° | randomized | 100% | 0% | 0% |
| best | slope_down | 20° | randomized | 97% | 2% | 0% |
| best | slope_down | 25° | randomized | 88% | 4% | 0% |

**final 的 15 cm／25° 全方向驗收：未達標。**

**best 的 15 cm／25° 全方向驗收：未達標。**

## 目標難度的失敗分類

下列數字為每 100 次測試的次數；接觸、姿態與越界原因可能同時發生，不能相加當作總失敗數。逾時指未通過且未被判定為跌倒或卡住。

| Checkpoint | 地形 | 擾動 | 接觸失敗 | 姿態失敗 | 越界 | 卡住 | 逾時 |
|---|---|---|---:|---:|---:|---:|---:|
| final | stairs_up | nominal | 0 | 0 | 0 | 7 | 0 |
| final | stairs_down | nominal | 0 | 0 | 0 | 2 | 0 |
| final | slope_up | nominal | 0 | 0 | 0 | 0 | 0 |
| final | slope_down | nominal | 0 | 0 | 0 | 0 | 0 |
| final | stairs_up | randomized | 2 | 0 | 0 | 3 | 0 |
| final | stairs_down | randomized | 2 | 7 | 0 | 0 | 0 |
| final | slope_up | randomized | 4 | 9 | 0 | 13 | 0 |
| final | slope_down | randomized | 0 | 0 | 0 | 0 | 7 |
| best | stairs_up | nominal | 0 | 0 | 0 | 1 | 0 |
| best | stairs_down | nominal | 0 | 0 | 0 | 0 | 0 |
| best | slope_up | nominal | 0 | 0 | 0 | 0 | 0 |
| best | slope_down | nominal | 0 | 0 | 0 | 0 | 0 |
| best | stairs_up | randomized | 4 | 1 | 0 | 0 | 0 |
| best | stairs_down | randomized | 1 | 7 | 0 | 0 | 0 |
| best | slope_up | randomized | 9 | 7 | 0 | 15 | 2 |
| best | slope_down | randomized | 0 | 4 | 0 | 0 | 8 |

首輪主要瓶頸是隨機化 25° 上坡。最後版本 100 次中有 13 次卡住、12 次失敗終止；最佳版本有 15 次卡住、12 次失敗終止、2 次逾時。兩者無擾動上坡均為 100%，顯示目前較弱的是擾動下的穩定性，而非 nominal 地形完全無法通過；摩擦、推擾、COM 等各自的因果影響尚未做消融。

下一輪建議先固定 25° 上坡與評估 seed，分別測試摩擦、推擾、COM、馬達／延遲的影響；增加高坡等級訓練覆蓋與擾動恢復訓練，再按接觸／姿態／卡住分類調整 reward。最佳 checkpoint 的定期選擇也應納入 randomized 目標測試，避免只依 nominal probe 選擇。這些調整尚未執行，未追加第二輪訓練。

## CPU sim2sim 影片

- stairs_up 2 cm：success；進度 3.56/3.55 m，時間 10.28 秒。
- stairs_up 15 cm：success；進度 3.55/3.55 m，時間 10.38 秒。
- stairs_down 2 cm：success；進度 3.55/3.55 m，時間 10.76 秒。
- stairs_down 15 cm：success；進度 3.56/3.55 m，時間 11.80 秒。
- slope_up 10°：success；進度 4.80/4.80 m，時間 11.56 秒。
- slope_up 25°：success；進度 4.80/4.80 m，時間 11.12 秒。
- slope_down 10°：success；進度 4.81/4.80 m，時間 14.90 秒。
- slope_down 25°：success；進度 4.80/4.80 m，時間 17.54 秒。

GPU 隨機化 25° 上坡未通過影片：`gpu_randomized_slope_up_25_failure_trial2.mp4`。使用最後 checkpoint、seed 2026、額外 100 個環境錄下 trial 2（失敗終止）。此錄影批次成功率為 75%，數據在 `gpu_failure_recording.json`；正式驗收仍以 `final_evaluation.json` 為準。CPU nominal 成功影片使用不同擾動條件。

成功是 20 秒內到達完整地形出口，且未發生失敗終止。卡住為連續 3 秒前進不足 10 cm；存活不代表通過。

完整結果、TorchScript、影片與 log：`artifacts/2026-10-04_19-22-18_first_round`。CPU nominal 與 GPU randomized 影片分別標示；Mac 原生 viewer 尚未在 Mac 本機驗證。

工程驗證：21 項單元測試與 1 項 GPU 整合測試通過；TorchScript 重新載入後，單一與多環境輸出和原推論一致。訓練／部署共用關節、站姿、硬角度限制、PD 與扭矩限制；CPU 的積分器、接觸力門檻和 world-command 轉換已對齊。測試 logs 保存在 artifacts。
