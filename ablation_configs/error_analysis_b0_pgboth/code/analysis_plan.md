# 预注册：三折新 B0 与 PG-both 离线误差分析口径与判据

本文件在**首次运行前冻结**，其 sha256 记入每次 `results/run_00X/manifest.json`。
分析脚本 `error_analysis.py` 的常量必须与本文件一致；如需变更判据，另建新文件与新 run，不改本文件。

## 0. 数据与边界

- 输入：六组 `predictions.bbox.json`（fold6/7/8 × b0/pg_both），各自目录内的 `test.json`（六个字节相同，
  sha256 `19aa601904243be548c1551b247f331f1317dc6a9bf4ac9911e6b07398ca919f`）、`metrics.json`、`metadata.json`。
- 冻结后处理：rcnn `score_thr=0.05`、`nms='nms'`、`iou_threshold=0.5`、`max_per_img=100`；
  rpn `iou=0.7`、`max_per_img=1000`；`inference_on='teacher2'`。
- 232 图 / 546 非 crowd 标注 / 单类别 id 0；全部 `iscrowd=0`。
- 两个尺寸口径必须分开标注：`ann.area`（分割多边形面积，**COCOeval `areaRng` 用它**）→ 416 / 128 / 2；
  bbox `w*h` → 339 / 188 / 19。**不对 APl / `AR_l@1000` 下任何结论**（COCO 口径 large 仅 2 个 GT）。
- 本轮**不复用**该 232 图作为独立验证集：它此前已用于 AP75 讨论，本轮结论只是诊断线索。
- `ShipRSImageNet` 按 `audit_softnms_cr4o9epx/CORRECTION.md` 实为**光学遥感、非 SAR**，其"独立 SAR"表述已撤回；
  ShipRSImageNet test 保持封存，不得当作独立 SAR 验证复用。

## 1. 常量

```
IOU_THR = 0.5            # 冻结贪心口径
AP75_IOU = 0.75          # 定位质量阈
NEAR_IOU = 0.3           # 漏检细分"近邻"切点（扩展量）
BANDS = [(0.9,1.01),(0.8,0.9),(0.7,0.8),(0.6,0.7),(0.5,0.6)]
PERM_SEED = 0 ; N_PERM = 10000
FORBIDDEN_IMPORTS = {ssod, mmdet, mmcv, mmcv_full, torch}
```

## 2. 三套计数命名空间（禁止相加、禁止互相比较）

| 命名空间 | 语义 | 匹配器 |
|---|---|---|
| `legacy_*` | 冻结贪心口径：组内按分数降序、每条预测取单一最大 IoU GT、一对一占用 | 复刻 `score_thr_bkg.py:33-94` |
| `coco_*` | COCOeval 的 TP/FP/ignore 分解与 AP/AR | pycocotools 自有 IoU（含 `+1`）与 `areaRng` |
| `coverage_*` | 逐 GT 覆盖视图（`best_iou_any`、`covered_at_50/75`、`miss_*`、`imprecise_*`） | 由 `legacy` 的占用 + 逐 GT 任意框最大 IoU 组合 |

`legacy_*` 是唯一与冻结脚本做影子对照的一套（自检靶面）。

## 3. 并列打破规则（写死）

- 预测排序：Python `sorted` 稳定排序（同分保持文件原序）。
- GT argmax：严格 `v > best_iou`，同 IoU 取最小 GT 下标（标注 id 升序）。
- 上述与 `score_thr_bkg.py:51,57` 一致，不得改动。

## 4. 派生量定义

- `miss_total`：未被 `legacy` 占用的 GT 数。
- 漏检细分（用 `best_iou_any`）：`≥0.5` → `miss_stolen`；`[NEAR_IOU,0.5)` → `miss_near`；否则 `miss_none`。
  并列报 `miss_best_iou_any_bins`：漏检 GT 的 `best_iou_any` 在
  `[0,0.1)/[0.1,0.2)/[0.2,0.3)/[0.3,0.4)/[0.4,0.5)` 五档的直方图（**无切点**），
  使"近邻漏检"的量级不挂在自定的 `NEAR_IOU` 上。
- `imprecise_total`：已占用且 `matched_pred_iou < 0.75`（用**被分配**的那个框的 IoU）。
- `n_preds_iou_ge_0.5`：该图对某 GT 的 IoU ≥ 0.5 的预测条数。
- 每个派生量在报告中标明用的是 `matched_pred_iou` 还是 `best_iou_any`。

## 5. Part A 判据（共同瓶颈）

对 fold `f∈{6,7,8}`、模式 `m∈{bkg,dupe,miss,imprecise}` 计 `n_m(f)` 与占比
（`s_bkg=n_bkg/num_preds`、`s_dupe=n_dupe/num_preds`、`s_miss=n_miss/546`、`s_imp=n_imp/546`），
以及逐图向量 `v_m(f)∈Z^232`。**判定 m 为共同瓶颈需五条同时成立：**

1. 排序稳定：按计数的前两名在三折完全相同。
2. 压倒性：`n_bkg/(n_bkg+n_miss+n_imp) ≥ 0.8` 在每一折成立（仅对 bkg 适用）。
3. 逐图一致：`v_m` 三个折两两 Pearson 全部 > 0.5，且三折 top-quartile 硬图交集 ≥ 15
   （随机水平 = 232·(58/232)³ ≈ 3.625）。
4. 非离群驱动：top-10 图对 `n_m` 的贡献每折 < 40%。
5. 分层稳定（仅 bkg）：小框占比与 `≥0.9` 占比在各折间极差 < 10 个百分点且偏斜方向一致。

**判定"折特有"**：某模式只在单折进前二，或它与其余折的逐图 Pearson < 0.3。
**允许不裁定**：任一条不满足即如实写"未形成唯一共同瓶颈"；不得为凑出唯一瓶颈而修改阈值或追加训练。
**反证条件**：(a) 首位模式跨折变化；(b) 任一折对 `r_bkg < 0.5`；(c) 三折 top-quartile 交集接近随机；
(d) `n_bkg` 在某折由少数图主导；(e) bkg 的小框/高分偏斜方向不一致。

## 6. Part B 判据（配对与裁定）

对每折取 B0 / PG-both 的逐 GT `best_iou_any`，`Δiou = pg − b0`。
翻转集：`gain_thr = {b0 < thr ≤ pg}`、`loss_thr = {pg < thr ≤ b0}`，`thr ∈ {0.5, 0.75}`。
特征：`ann_area`、bbox 面积、COCO 分箱、`nn_dist_over_sqrt_area`（GT 中心最近邻 / √ann_area）、
`best_score_b0/pg`、`contested = (n_preds_iou_ge_0.5_b0 ≥ 2) or (n_preds_iou_ge_0.5_pg ≥ 2)`。
检验：面积与聚集度两特征各做中位数差的置换检验（`N_PERM=10000`，`PERM_SEED=0`），并报 Cliff's δ。

**裁定规则（同一函数、同一阈值，应用于全部比较对）**：

- **同一现象（反向）** ⟺ 两特征都 `p > 0.05` 且 `|δ| < 0.2`，且 churn 近似镜像：
  `|A|/|B| ∈ [0.67, 1.5]` 且 `|gain_thr(fA)|/|loss_thr(fB)| ∈ [0.67, 1.5]`。
- **两个不同现象** ⟺ 任一特征 `p < 0.05` 且 `|δ| ≥ 0.33`，或两组比值都超出 `[0.67, 1.5]`。
- **功效不足、不裁定** ⟺ 其余情况（翻转集仅数 10 个目标，不得放宽口径凑结论）。

应用的比较对：(a) **主比较** `loss75(fold7)` vs `gain75(fold8)`；
(b) 三折各自的内部对 `loss75(f)` vs `gain75(f)`。fold6 的裁定**标注参考**，不构成验证证据（它已被用于选方案）。
结论措辞上限为"与…一致 / 在本样本量下无法区分"。

## 7. Part C（证据表，不选模块）

候选方向按固定模式顺序（bkg → miss → imprecise → dupe）呈现，每条同时列**支持证据**与**反证**。
脚本**不做排序、不选模块、不强排方向**；允许"不裁定机制"。不提出任何在本测试集上调 score/NMS 阈值的方案，
不叠加 M2/FG。

## 8. 成因字段边界

"未产生好框 / 好框被分数过滤 / 好框被 NMS 抑制"三项本轮**一律记"未知"**，
不得由低重复率或低覆盖率倒推。候选级归因属阶段二，本轮不执行。

## 9. 交付达成条件

六组身份与指标复算通过；输出能区分匹配（`legacy`）、几何覆盖（`coverage`）、分数排序（score bands）；
候选表同时呈现支持与反证，允许不裁定或不强排方向。
