# 第二轮：两个新增折（fold7、fold8）的新 B0 与 PG-both（2026-09-26 启动）

本轮是**两个新增折、四次训练**（不是"四折"）：只扩折验证 fold6 里唯一为正的 PG-both；
fold6 材料见 `fold6_seed678/ablation_results.md`，**不重跑**。
每组材料保存在各自 suite 目录（`fold7_seed678/`、`fold8_seed678/`），本文件是跨折索引与汇总。

## 1. 用户 2026-09-26 给定的本轮范围（照此执行）

- **两个新增折、共 4 次训练**：`fold7 B0 → fold7 PG-both → fold8 B0 → fold8 PG-both`。全部独立初始化；
  **不根据 fold7 增益正负决定是否继续 fold8**；发生训练或验收失败时**暂停处理**。
- **冻结**：沿用本轮代码、PG 曲线、seed 678、32000 步与 `--no-validate`，不调参；
  每折使用自己的 3 张标注 SAR 图及对应 Phase2 权重，Phase1 共用；同折 B0 与 PG-both 除模块开关及输出标识外保持一致。
- **汇总口径（预先固定）**：每折 `ΔmAP = PG-both − 同折新 B0`；
  **主要看新增 fold7/8 的逐折增量与平均增量**（因 fold6 已用于选择方案），同时报告三折绝对指标、逐折增量与三折平均增量；
  AP75、APs、AR@100 一并列出。旧失效实验不参与。
- **完成四次训练与评估后再决定下一轮**；期间不加组合模块；完成组按已批准的清理规则处理。

## 2. 材料生成（2026-09-26 13:42，退出码 0，未训练）

```bash
# 逐折各执行一次，cwd = checkout 根
PYTHONPATH=$PWD python tools/prepare_ablation_suite.py configs/reproduce/phase3_dual_teacher_ssdd.py \
  --out-dir ablation_configs/fold<F>_seed678 --work-root work_dirs/ablation_v1/fold<F>_seed678 \
  --seed 678 --cfg-options fold=<F> percent=3
```

| 折 | suite | work root | 生成日志 / 退出码 | `b0.py` sha256（前 16） | `pg_both.py` sha256（前 16） |
|---|---|---|---|---|---|
| 7 | `ablation_configs/fold7_seed678/` | `work_dirs/ablation_v1/fold7_seed678/` | `suite_generation.log` / `suite_generation_exit_code.txt` = 0 | `3cb1f80d66234345` | `acf0022911c7b095` |
| 8 | `ablation_configs/fold8_seed678/` | `work_dirs/ablation_v1/fold8_seed678/` | 同上 = 0 | `05aed2213033f111` | `624d280a3d3fcf97` |

**注意生成器本身不做源码断言**（它只 import `ssod.utils.ablation`），故生成时用 `PYTHONPATH=$PWD` 显式钉住本 checkout；
本轮训练仍走 `tools/train_ablation.py` 的 `pin_repository()` + `source_manifest()`（见下）。

## 3. 与 fold6 的一致性证据（逐项核对过）

- **训练导入的 9 个模块哈希与 fold6 逐模块相同**（fold7 `b0_source_receipt.json` ↔ fold6 同名文件；`repo_root` 亦相同）。
- B0 模板 sha256 相同：`cc41b5d09730d4a5cc137ad7dcbb7fc4d2601260ebccee4591b0df6ba4d284bc`（两折 manifest 的 `baseline_sha256`）。
- Phase1 共用且实测哈希相同：`70627d7eee2a55f5…`（`work_dirs/phase1_pretrain_optical/100/1/iter_8000.pth`）。
- test.json 相同：`19aa60190424…`（232 图冻结测试集）。
- **每折 Phase2 不同**（这是"每折自己的 Phase2 权重"的落点）：fold7 `c97550fd0e05070f…`、
  fold8 `45cc4962dc64c74f…`，路径 `work_dirs/phase2_pretrain_optical_sar/3/<fold>/iter_11200.pth`。
- **每折标注不同**：fold7 `instances_train2017.7@3.json` `622326f60aa3f1ae…`、
  fold8 `instances_train2017.8@3.json` `6870197741d60ca3…`（各自 `-unlabeled` 见 manifest）。
- 配置差异已逐字 diff：
  `fold<F>/b0.py` 相对 `fold6_seed678/b0.py` 只差 5 处（`load2_from`、两个 ann_file、`work_dir`、`fold`）；
  `fold<F>/pg_both.py` 相对同折 `b0.py` 只差 3 处（新增 `pg` 块、`work_dir`、`ablation_experiment`）。
  → 满足"同折 B0 与 PG-both 除模块开关及输出标识外保持一致"。

## 4. 进度（逐步更新，失败即停）

| 步骤 | 状态 | 关键记录 |
|---|---|---|
| fold7 材料生成 | 完成 | 见第 2 节 |
| fold7 B0 初始化预检（CPU） | 完成，退出码 0 | `fold7_seed678/b0_init.log`：四分支各 654 张量 strict；T1=S1、T2=S2、T1!=T2；load2 实测为 `…/3/7/iter_11200.pth` |
| fold7 B0 权重哈希实算 | 完成 | `fold7_seed678/b0_initialization_sha256.json`（两项 `matches_manifest_record: true`） |
| fold7 B0 训练 | 完成，退出码 0 | 13:44:14 → 22:33:39（8.82 h）；末步 `Iter [32000/32000]`；命令逐字见 `fold7_seed678/b0_launch_record.json` |
| fold7 B0 完成判定 | 完成 | 退出码 0；`iter_32000.pth` size 1448209105，sha256 `5bc520ed1d0e43ad009f26a41b78fa88761cab3e3f6bbde6722459895646384d`；`latest.pth` 实测同 SHA |
| fold7 B0 异常分项（三类分开） | 完成 | ①非有限 loss **0** 条（按 `key: value` 字段解析；初版整行子串匹配误得 13，已改正并记入完成记录）②`grad_norm: inf` 采样窗口 **13/640**，出现迭代 50/750/3100/6200/8450/11700/14050/16800/19300/23350/25500/27200/31400；这 13 行的 `loss` 字段实测全部有限（0.57–1.23）③traceback **0**。`[PG weights]` 0 行；run `.log.json` 中 `bbox_mAP` 0 处 |
| fold7 B0 跳步区间（推导） | 完成 | checkpoint `meta.fp16.loss_scaler`：scale 32768（初值 65536，净一次减半）、growth_interval 2000、`_growth_tracker` 626 → `1 ≤ 跳步数 ≤ 17`（**区间，非计数**；13 只是采样窗口数） |
| fold7 B0 评估 | 完成，退出码 0 | `--eval … --fold 7 --out-dir fold7_seed678/b0_evaluation`；232/232 覆盖；**mAP 0.501 / AP50 0.829 / AP75 0.574 / AR@100 0.583**，APs 0.543、APm 0.371、APl 0.042 |
| fold7 B0 权重清理 | 完成 | `fold7_seed678/deletion_manifest_20260926.json`：删 iter_4000..28000 与 `latest.pth`（独立副本、非硬链接，先验同 SHA），释放 11.59 GB；保留 `iter_32000.pth` |
| fold7 PG-both 预检+验收 | 完成 | `pg_both_init.log` exit 0，四分支各 654 张量 strict，load2 为 `…/3/7/iter_11200.pth`；`pg_both_acceptance/result.json` **status=passed**、`loss_routing_verified: true`、`config_sha256` = `acf0022911c7b095…` 与 `pg_both.py` 一致 |
| fold7 PG-both 训练 | 完成，退出码 0 | 22:38:58 → 07:24 左右（8.77 h）；末步 `Iter [32000/32000]`；命令逐字见 `pg_both_launch_record.json`；`iter_32000.pth` sha256 `140d987d56f44f9a…`，`latest.pth` 实测同 SHA |
| fold7 PG-both 异常分项（三类分开） | 完成 | ①非有限 loss **0** 条（按字段解析）②`grad_norm: inf` 采样窗口 **15/640**，迭代 50/550/2600/4850/8700/11050/13150/16100/20100/22150/24350/26400/28700/28900/31200 ③traceback **0**。`[PG weights]` 641 行（恰为 iter1..32000 的采样点） |
| fold7 PG-both 跳步区间（推导） | 完成 | `meta.fp16.loss_scaler`：scale 16384（初值 65536，净两次减半）、`_growth_tracker` 838 → `2 ≤ 跳步数 ≤ 18`（**区间**；15 只是采样窗口数） |
| fold7 PG-both 全日志 PG 核验 | 完成，退出码 0 | `pg_both_pg_weights_verification.log`：**PASS**，641 行全部与闭式公式一致；first `sup2=0.1 unsup2=0.2`、last `sup2=0.2 unsup2=0.4`；`max |deviation| = 4.995e-10`。与 fold6 E4 在 iter 2950/8950/11650 上逐位相同 → 冻结曲线一致 |
| fold7 PG-both 评估 | 完成，退出码 0 | `--eval pg_both.py <ckpt> --fold 7 --out-dir fold7_seed678/pg_both_evaluation`；232/232 覆盖；**mAP 0.481 / AP50 0.828 / AP75 0.517 / AR@100 0.576**，APs 0.526、APm 0.358、APl 0.060 |
| fold8 B0 预检 / 收据 / 哈希 | 完成 | `b0_init.log` exit 0，四分支各 654 张量 strict，load2 为 `…/3/8/iter_11200.pth`；`b0_source_receipt.json` 9 模块哈希与 fold7 **逐模块相同**；`b0_initialization_sha256.json` 两项 `matches_manifest_record: true` |
| fold8 B0 训练 | 完成，退出码 0 | 07:30:30 → 16:23:35（8.91 h）；末步 `Iter [32000/32000]`；`iter_32000.pth` sha256 `a30e5a317f8e5255…`，`latest.pth` 实测同 SHA |
| fold8 B0 异常分项（三类分开） | 完成 | ①非有限 loss **0** 条（按字段解析）②`grad_norm: inf` 采样窗口 **14/640** ③traceback **0**；`[PG weights]` **0** 行（无 PG hook，符合预期） |
| fold8 B0 跳步区间（推导） | 完成 | `meta.fp16.loss_scaler`：scale 32768（初值 65536，净 1 次减半）、`_growth_tracker` 749 → `1 ≤ 跳步数 ≤ 17`（**区间**；14 只是采样窗口数） |
| fold8 B0 评估 | 完成，退出码 0 | `--eval b0.py <ckpt> --fold 8 --out-dir fold8_seed678/b0_evaluation`；232/232 覆盖；**mAP 0.441 / AP50 0.811 / AP75 0.438 / AR@100 0.563**，APs 0.471、APm 0.361、APl 0.108（APl 仅 2 个目标，不可解释） |
| fold8 B0 权重清理 | 完成 | `fold8_seed678/deletion_manifest_20260927.json`：删 iter_4000..28000 与 `latest.pth`（先验同 SHA），释放 11.59 GB；保留 `iter_32000.pth` |
| fold8 PG-both 预检+验收 | 完成 | `pg_both_init.log` exit 0（load2 = `…/3/8/iter_11200.pth`）；`pg_both_acceptance/result.json` **status=passed**、`loss_routing_verified: true`、`config_sha256` = `624d280a3d3fcf97…` 与 `pg_both.py` 一致 |
| fold8 PG-both 训练 | 完成，退出码 0 | 16:27:42 → 01:20:07（8.87 h）；末步 `Iter [32000/32000]`；`iter_32000.pth` sha256 `5c223a7c6b266420…` = `latest.pth` |
| fold8 PG-both 异常分项（三类分开） | 完成 | ①非有限 loss **0** 条（按字段解析）②`grad_norm: inf` 采样窗口 **14/640** ③traceback **0**；`[PG weights]` **641** 行。跳步区间：scale 32768（净 1 次减半）、`_growth_tracker` 1140 → `1 ≤ B ≤ 17`（区间） |
| fold8 PG-both 全日志 PG 核验 | 完成，退出码 0 | **PASS**：641 行与闭式一致；first `sup2=0.1 unsup2=0.2`、last `sup2=0.2 unsup2=0.4`；`max |deviation| = 4.995e-10` |
| fold8 PG-both 评估 | 完成，退出码 0 | `--fold 8`；232/232 覆盖；**mAP 0.462 / AP50 0.839 / AP75 0.475 / AR@100 0.567**，APs 0.493、APm 0.372、APl 0.120 |
| fold8 PG-both 权重清理 | 完成 | `fold8_seed678/pg_both_deletion_manifest_20260928.json`：释放 11.59 GB；保留 `iter_32000.pth` |
| 三折汇总 | 完成 | 见第 5 节（主结果先报新增两折） |

**启动时的已知现象（不构成停训条件，2026-09-26 口径统一后照此记录）**：fold7 B0 首条日志 `Iter [50/32000]` 即含 `grad_norm: inf`，
这与 fold6 五组全部出现该字段一样，**只说明是跨组的共同观察**；把它归因为"配方级属性"**尚未证实**
（本轮不做成因实验）。**沿用既定处理：不因单次出现就停训**；同时每折每组按下列三类**分开统计**，不合并、不互相替代：

1. **非有限 loss**：日志/`.log.json` 中 `loss` 或各 `*_loss*` 键出现 `nan`/`inf` 的条目数；
2. **采样到的梯度 Inf**：`grad_norm: inf` 的**窗口数**——**不能称为实际跳步率**（`log_interval=50` 只覆盖被打印的采样点），
   跳步数只能用 `scale` / `_growth_tracker` 约束成**区间**；
3. **异常退出**：训练进程实际退出码与 traceback。

**禁止写法**：不写 `"nonfinite": 0` 之类的合并字段，也不写"非有限统计全零"。

## 5. 结果表（Δ 一律由各组 `metrics.json` 存储值直接相减，不做二次舍入）

**汇报顺序（用户 2026-09-27 指令，固定）**：

1. **主结果**：新增 fold7/8 的**逐折 ΔmAP** 与**两折平均 ΔmAP**；
2. **补充结果**：含 fold6 的**三折**绝对指标、逐折 Δ、三折平均 Δ；
3. 两部分**都并列 AP75、APs、AR@100**。

**理由（用户原话要点）**：fold6 已被用于选择 PG-both，**不能只报三折平均让它的正增益掩盖新增两折的验证结果**。
fold7 的负结果（ΔmAP −0.020）**完整保留，不据此中途调参**；是否继续推进 PG-both 等 fold8 配对结果齐全后再定。

### 主结果：新增两折（fold7、fold8）

| 折 | B0 mAP | PG-both mAP | **ΔmAP** | B0 AP75 | PG AP75 | ΔAP75 | B0 APs | PG APs | ΔAPs | B0 AR@100 | PG AR@100 | ΔAR@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 0.501 | 0.481 | **−0.0200** | 0.574 | 0.517 | −0.0570 | 0.543 | 0.526 | −0.0170 | 0.583 | 0.576 | −0.0070 |
| 8 | 0.441 | 0.462 | **+0.0210** | 0.438 | 0.475 | +0.0370 | 0.471 | 0.493 | +0.0220 | 0.563 | 0.567 | +0.0040 |
| **两折平均 Δ** | — | — | **+0.000500** | — | — | −0.010000 | — | — | +0.002500 | — | — | −0.001500 |
| 同两折附加 | AP50 平均 Δ +0.013500 | APm 平均 Δ −0.001000 | | | | | | | | | | |

**主结果读数**：新增两折的 ΔmAP 一负一正（−0.0200 / +0.0210），**平均 +0.0005**，
即 PG-both 在未参与选方案的折上**未提供明确的净增益证据**；AP75 平均 Δ −0.0100、AR@100 平均 Δ −0.0015 亦为负，APs 平均 +0.0025 接近零。
注意措辞边界：平均增益**明显小于**单折增量的量级，不能表述为"与折间波动同阶"；两折也确实不足以确定总体效应。

### 补充结果：含 fold6 的三折

| 折 | B0 mAP | PG-both mAP | ΔmAP | ΔAP75 | ΔAPs | ΔAR@100 |
|---|---|---|---|---|---|---|
| 6（参考，已用于选择方案） | 0.483 | 0.517 | +0.0340 | +0.0830 | +0.0360 | +0.0240 |
| 7 | 0.501 | 0.481 | −0.0200 | −0.0570 | −0.0170 | −0.0070 |
| 8 | 0.441 | 0.462 | +0.0210 | +0.0370 | +0.0220 | +0.0040 |
| **三折平均 Δ** | — | — | +0.011667 | +0.021000 | +0.013667 | +0.007000 |
| **三折平均绝对值** | B0 0.475000 | PG-both 0.486667 | — | — | — | — |

附加：三折 AP50 平均 Δ +0.013667；APm 平均 Δ +0.006667。
（fold6 的逐项见其台账 `fold6_seed678/ablation_results.md`；三折等权平均按逐折配对 Δ 再平均，不因某折绝对指标弱而删折。）

**注**：上表 Δ 全部由各组 `ablation_configs/fold<N>_seed678/<组>_evaluation/metrics.json` 的存储值直接相减，
未做二次舍入；三折平均绝对值由存储值平均得到（B0 0.475000 / PG-both 0.486667）。

**APl / AR_l 的可解释性限制（实测样本量）**：`data/ssdd/annotations/test.json` 非 crowd 标注共 546，
按 COCO 面积阈值为 small 416 / medium 128 / **large 2**。故 APl（fold7：B0 0.042 → PG-both 0.060）
与 `bbox_AR_l@1000`（0.200 → 0.450）都只建立在 **2 个目标**上，波动不可解释，**不作为本折结论依据**。
本折结论只用 mAP / AP75 / APs / AR@100（及 APm）这些样本量足够的量。

## 6. 本轮结论（2026-09-28 用户裁定：正式收口）

**统一结论（用户给定措辞，台账与后续材料照此引用）**：

> PG-both 在新增两折上的 ΔmAP 分别为 −0.020、+0.021，平均 +0.0005，未提供明确的净增益证据。
> 包含方案选择折 fold6 后，三折平均为 +0.011667，主要由 fold6 贡献。当前冻结配方暂不继续扩折或扩种子。

**决定**：**停止追加当前 PG-both 配方的训练**；保留代码、最终权重与全部记录；
**暂不将 PG-both 纳入默认方案**。

**决策依据（用户给出）**：新增两折平均 mAP 只增加 **0.05 个百分点**，同时 AP75 平均下降 **1.0 个百分点**、召回略降；
现有结果不足以支持继续投入成对长训。**这是投入优先级的决定，不是证明 PG-both 无效**——两折确实不足以确定总体效应。

**下一步（用户 2026-09-28 指定，先分析不训练）**：
1. 以**三折新 B0** 为基础，统计背景误检、重复检测、漏检与定位误差，寻找**共同瓶颈**；
2. 对照 PG-both，重点看 fold7 的 AP75 下降与 fold8 的提升分别落在**哪些目标**上；
3. 依共同瓶颈再选**一个**修改方向。当前**不直接叠加 M2 / FG**，也**不据这些测试集结果调整后处理阈值**。
