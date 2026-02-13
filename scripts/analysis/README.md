# 数据分析与可视化（Uncertainty Research）

本目录 `uncertainty_research/scripts/analysis/` 提供一套**可复用**的分析/可视化脚本，用于处理 early-exit 实验产出的 `early_exit_generic_shard*.csv`，并生成 OOD 检测表格、分布图、动力学曲线、校准热力图与极端样本可视化。

---

### 0. 你需要先知道的目录约定

实验输出（由 launcher 生成）通常长这样：

- **原始分 shard 结果**：
  - `{OUT_ROOT}/{RUN_ID}_{label}_{severity}_{TAG}/shard_{i}/early_exit_generic_shard{i}.csv`
  - 其中 `label` 常见为：`none / noise / blur / go_stanford`（由你的启动脚本决定）

分析输出统一写到：

- **分析根目录**：`{OUT_ROOT}/{RUN_ID}_analysis_{TAG}/`

`common_io.py` 默认会用下面这个 glob 来找原始 CSV（如果你不手动指定 `--input_glob`）：

```text
{OUT_ROOT}/{RUN_ID}_*_{TAG}/shard_*/early_exit_generic_shard*.csv
```

---

### 1. 最推荐的一条“标准流水线”

只要你跑完实验（生成了若干 shard CSV），分析通常按这个顺序走：

- **Big Table（同时生成 `all_concat.csv`）**
- **Histogram（读 `all_concat.csv`）**
- **AUROC dynamics（读 `big_table_long.csv`）**
- **Raw dynamics（读 `all_concat.csv`）**
- **Calibration Kendall（读 `all_concat.csv`）**
- **Render extremes（读 `all_concat.csv` + 需要模型权重/vae）**

下面每个脚本都给出可复制命令与输出文件说明。

---

### 2. Big Table：全组合 OOD 检测表（AUROC/AUPR/FPR95）

脚本：`big_table.py`

**用途**

- 读取所有 shard 的 `early_exit_generic_shard*.csv`
- 以 **ID = `recon/none/0`** 为基准
- 自动枚举 OOD：
  - `recon/noise/k`
  - `recon/blur/k`
  - `go_stanford/none/0`（若数据存在）
- 对每个 `probe_step`、每个 method（Pixel/Latent/LPIPS/DreamSim）计算：
  - **AUROC / AUPR / FPR95**
- 同时在 analysis 根目录写出一份**全量聚合原始数据**：`all_concat.csv`

**启动命令**

```bash
cd /home/payneli/project/nwm
/home/payneli/anaconda3/envs/wm/bin/python -u uncertainty_research/scripts/analysis/big_table.py \
  --out_root "/home/payneli/project/nwm/uncertainty_research/results" \
  --run_id "<RUN_ID>" \
  --tag "<TAG>" \
  --probe_steps "2,5,10,20,30,40,50"
```

你也可以用目录里的快捷脚本（等价）：

```bash
bash /home/payneli/project/nwm/uncertainty_research/scripts/analysis/run_big_table.sh
```

> 注意：`run_big_table.sh` 里需要你把 `<RUN_ID>`、`<TAG>` 替换成真实值。

**输出（会写到）**

- `{OUT_ROOT}/{RUN_ID}_analysis_{TAG}/all_concat.csv`（**强烈建议保留**，后续脚本都会优先读它）
- `{OUT_ROOT}/{RUN_ID}_analysis_{TAG}/big_table/`
  - `big_table_long.csv`（长表：每行是一个 (ood, step, method)）
  - `big_table_auroc.csv`
  - `big_table_aupr.csv`
  - `big_table_fpr95.csv`
  - `big_table_auroc_step{final_step}.csv`（final step 的 AUROC “paper view”）

---

### 3. Score Distribution Histogram：分数分布直方图

脚本：`score_histogram.py`

**用途**

- 画 ID vs OOD 的分数分布
- 支持两种布局：
  - `overlay`：一个 panel 里叠很多组（可能颜色混叠）
  - `compare2`：**每个指标一行**、两列（ID vs Stanford、ID vs Noise-5），每个 panel 只叠 2 组，视觉更干净
- 支持 `--all_steps`：对所有 `probe_step` 批量出图
- 图标题/文件名会对 `lpips/dreamsim` 使用 **`*_var`** 展示名（更一致）

**启动命令（推荐：compare2 + all_steps）**

```bash
cd /home/payneli/project/nwm
/home/payneli/anaconda3/envs/wm/bin/python -u uncertainty_research/scripts/analysis/score_histogram.py \
  --out_root "/home/payneli/project/nwm/uncertainty_research/results" \
  --run_id "<RUN_ID>" \
  --tag "<TAG>" \
  --layout compare2 \
  --score_cols "lpips,pixel_var,latent_var,dreamsim" \
  --compare_oods "stanford,noise5" \
  --style filled \
  --bins 50 \
  --all_steps
```

**输出（会写到）**

- `{OUT_ROOT}/{RUN_ID}_analysis_{TAG}/score_histograms/`
  - `hist_compare2_<metrics>_step{step}.png`（每个 step 一张）
  - 如果你用 `overlay`/`multi` 布局，会输出 `hist_*.png` 或 `hist_multi_*.png`

---

### 4. Discriminability Dynamics：AUROC/AUPR/FPR95 vs Probe Step

脚本：`auroc_dynamics.py`

**用途**

- 读取 `big_table/big_table_long.csv`
- 画（AUROC 或 AUPR 或 FPR95）随 `probe_step` 的变化曲线
- 默认 x 轴是正向（2→50）；可选 `--invert_x` 变成 50→2
- 支持 `--grid`：把多种方法画在同一张 multi-panel 图里
- 会把用于绘图的表格数据同时导出（便于你后续论文作图/复现）

**启动命令（推荐：grid）**

```bash
cd /home/payneli/project/nwm
/home/payneli/anaconda3/envs/wm/bin/python -u uncertainty_research/scripts/analysis/auroc_dynamics.py \
  --out_root "/home/payneli/project/nwm/uncertainty_research/results" \
  --run_id "<RUN_ID>" \
  --tag "<TAG>" \
  --metric auroc \
  --methods "Pixel_Var,Latent_Var,LPIPS_Var,DreamSim_Var" \
  --grid
```

**输出（会写到）**

- `{OUT_ROOT}/{RUN_ID}_analysis_{TAG}/auroc_dynamics/`
  - `{metric}_vs_step_long.csv`（例如 `auroc_vs_step_long.csv`）
  - `{metric}_vs_step_wide.csv`
  - `{metric}_vs_step_grid.png`（如果 `--grid`）
  - 或者每个方法一张：`{metric}_vs_step_{method}.png`

---

### 5. Raw Variance Dynamics：原始分数（归一化）随时间变化

脚本：`raw_dynamics.py`

**用途**

- 读取 `all_concat.csv`（若不存在会回退到 shard glob）
- 对每个 `(dataset, perturbation, severity)` 组，计算每个 step 的**均值**
- 支持 `--normalize minmax`（默认）：对每个指标在全局做 min-max，便于把不同组曲线放一张图比对

**启动命令**

```bash
cd /home/payneli/project/nwm
/home/payneli/anaconda3/envs/wm/bin/python -u uncertainty_research/scripts/analysis/raw_dynamics.py \
  --out_root "/home/payneli/project/nwm/uncertainty_research/results" \
  --run_id "<RUN_ID>" \
  --tag "<TAG>" \
  --metrics "lpips,pixel_var,latent_var,dreamsim" \
  --normalize minmax
```

**输出（会写到）**

- `{OUT_ROOT}/{RUN_ID}_analysis_{TAG}/raw_dynamics/`
  - `raw_dynamics_mean.csv`（用于绘图的聚合表）
  - `raw_variance_vs_step_grid.png`

---

### 6. Calibration：Kendall’s τ（含 p-value）热力图

脚本：`calibration_kendall.py`

**用途**

- 读取 `all_concat.csv`
- 在指定子集（filter）上，计算：
  - 行：`Pixel_Var / Latent_Var / LPIPS_Var / DreamSim_Var`
  - 列：`GT_MSE / GT_PSNR / GT_LPIPS / GT_DreamSim`
  - 单元格：**Kendall’s τ（tau-b）** 与 **p-value**
- 支持单个 `--filter` 或批量 `--filters`
- 支持 `--per_step_figs`：每个 step 单独出图（两张 heatmap：τ 与 p-value）
- 支持 `--invert_gt_psnr`：把 PSNR 变成“越大越差”的方向（用 `-PSNR`），更像误差指标

**启动命令（示例：对多个场景分别存子目录 + 每 step 出图）**

```bash
cd /home/payneli/project/nwm
/home/payneli/anaconda3/envs/wm/bin/python -u uncertainty_research/scripts/analysis/calibration_kendall.py \
  --out_root "/home/payneli/project/nwm/uncertainty_research/results" \
  --run_id "<RUN_ID>" \
  --tag "<TAG>" \
  --filters "recon:none:0,recon:noise:5,recon:blur:5,go_stanford:none:0" \
  --invert_gt_psnr \
  --per_step_figs
```

**输出（会写到）**

- `{OUT_ROOT}/{RUN_ID}_analysis_{TAG}/calibration_kendall/<dataset>_<perturbation>_<severity>/`
  - `kendall_tau_long.csv`（长表：含 `tau` 与 `p_value`）
  - `kendall_tau_step{step}.csv`（宽表 τ）
  - `kendall_p_step{step}.csv`（宽表 p-value）
  - `kendall_step{step}_tau_p.png`（如果 `--per_step_figs`）
  - 或 `kendall_tau_matrix_steps.png`（如果不加 `--per_step_figs`，会把多个 step 拼在一张图里）

---

### 7. Extreme Sample Visualization：极端样本渲染（定性图）

脚本：`render_extremes_by_condition.py`

**用途**

- 从 `all_concat.csv` 里按 `select_step`（通常 final step）挑：
  - 每个 condition、每个 metric 的 `max`/`min` 样本
- 对每个被选中的样本：
  - 保存 context（raw / perturb 后）
  - 对所有 probe_steps、所有 seeds 生成预测并排图
- 默认（`--summary_layout horizontal`）是“横向布局/转置布局”：
  - 第一列：`context 1..T + ground truth`
  - 后续每列：`probe_step=xx` +（可选）`GT_LPIPS/GT_DreamSim/GT_PSNR/GT_SSIM`
- 旧布局（`--summary_layout vertical`）：
  - 第一行：`context 1..T + ground truth`
  - 后续每行：`probe_step=xx` +（可选）`GT_LPIPS/GT_DreamSim/GT_PSNR/GT_SSIM`
- 会同时写 `x0_summary.png` 和 `x0_summary_v2.png`（v2 用来规避 IDE 缓存）

**启动命令（示例：只渲染 recon none 的 lpips 最小值 1 个，用于验证）**

```bash
cd /home/payneli/project/nwm
/home/payneli/anaconda3/envs/wm/bin/python -u uncertainty_research/scripts/analysis/render_extremes_by_condition.py \
  --out_root "/home/payneli/project/nwm/uncertainty_research/results" \
  --run_id "<RUN_ID>" \
  --tag "<TAG>" \
  --model_path "<MODEL_PATH>" \
  --vae_path "<VAE_PATH>" \
  --device "cuda:0" \
  --diffusion_steps 50 \
  --go_eval_type full \
  --select_step 50 \
  --probe_steps "2,5,10,20,30,40,50" \
  --num_seeds 5 \
  --base_seed 1234 \
  --summary_layout horizontal \
  --deterministic_per_seed \
  --deterministic_perturb \
  --conditions "recon:none:0" \
  --metrics "lpips" \
  --extremes "min" \
  --limit 1
```

**输出（会写到）**

- `{OUT_ROOT}/{RUN_ID}_analysis_{TAG}/extremes_finalstep/`
  - `selected_maxmin.json`（记录选择的样本 idx 与数值）
  - `metric=<metric>/cond=<dataset>_<perturbation>_<severity>/<max|min>_idx<sample_idx>/`
    - `meta.json`（样本元信息）
    - `action.txt`
    - `context_raw.png`
    - `context_input.png`
    - `x0_x0_step{step}.png`（每个 step 一个 seed-row 的 grid）
    - `x0_summary.png`
    - `x0_summary_v2.png`

---

### 8. （可选）Small320 快速聚合分析

脚本：`uncertainty_research/scripts/analyze_small320_4gpu.py`

**用途**

- 这是一个“针对 small320 四个固定条件”的快速聚合脚本（不依赖 `analysis/` 框架）
- 会把四个固定输出拼起来，并计算简单均值与 AUROC（含 PSNR/SSIM 反向逻辑）

**启动命令**

```bash
cd /home/payneli/project/nwm
/home/payneli/anaconda3/envs/wm/bin/python -u uncertainty_research/scripts/analyze_small320_4gpu.py \
  --out_root "/home/payneli/project/nwm/uncertainty_research/results" \
  --run_id "<RUN_ID>" \
  --tag "<TAG>"
```

**输出（会写到）**

- `{OUT_ROOT}/{RUN_ID}_analysis_{TAG}/`
  - `all_concat.csv`
  - `means_by_condition_probe.csv`
  - `auroc_by_probe.csv`

配套等待脚本（会轮询 4 个输出 CSV 是否齐备，然后再跑上面的分析）：

- `uncertainty_research/scripts/wait_and_analyze_small320_4gpu.sh`

```bash
bash /home/payneli/project/nwm/uncertainty_research/scripts/wait_and_analyze_small320_4gpu.sh \
  --run_id "<RUN_ID>" \
  --out_root "/home/payneli/project/nwm/uncertainty_research/results" \
  --tag "<TAG>" \
  --sleep 60
```

---

### 9. 常见问题（FAQ）

- **Q: 为什么很多脚本要求 `--run_id` 和 `--tag`？**
  - **A**：它们共同决定分析根目录 `{RUN_ID}_analysis_{TAG}`，并用于定位原始 shard 输出的 glob 模式（见 `common_io.py`）。

- **Q: 我没有 `all_concat.csv`，能直接跑 histogram/raw_dynamics/calibration 吗？**
  - **A**：可以。它们会回退到按默认 glob 去读 shard CSV；但为了稳妥/复现，建议先跑一次 `big_table.py` 生成 `all_concat.csv`。

- **Q: `auroc_dynamics.py` 为什么找不到输入？**
  - **A**：它只读 `big_table/big_table_long.csv`，所以你必须先跑 `big_table.py`。



