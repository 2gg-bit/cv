# Soft-NMS 后处理替换 · 独立审计目录（本轮）

目标：不重训，只验证把最终检测后处理从 Hard NMS 换成 Linear Soft-NMS 能否提高 COCO AP。

**本目录是新增的独立产物**，不改动原实验材料（权重/配置/预测/日志/旧报告），不重训，不调阈值。

## 进度（对应 7 步计划）

- [x] 步骤 1：工程与环境核对（`conda dt`、git HEAD=f77f060、torch1.7/mmcv1.3.9/mmdet2.16.0）
- [x] 步骤 2：实验协议与权重清单（`protocol.json`、`model_manifest.json`）
- [x] 步骤 3：独立开发集 —— HRSID 未就绪，改用 **ShipRSImageNet 官方 val（550 图）**，
      已核验不在训练数据中（训练用 DIOR+SSDD）；Ship-only（DOCK 排除）。见 `dev_manifest.json`。
      ⚠️ ShipRSImageNet 为**光学遥感**（非 SAR），仅作外部光学域对照；独立 SAR 验证仍待 HRSID，见 `CORRECTION.md`。
- [x] 步骤 4：候选缓存 + 原 Hard NMS 回放验收（三 fold 逐框位级一致，见 `step4_acceptance.md`）
- [x] 步骤 5：Linear Soft-NMS 接入 + 行为验收（`eval_ab.py`；空/单/同分候选、新分数、无额外过滤）
- [x] 步骤 6：开发集三组 A/B 离线评估（见 `dev_ab_report.md`）
- [ ] 步骤 7：封存测试集最终评估（**待用户按 dev_ab_report 结论决定是否进入**）

## 文件清单

| 文件 | 说明 |
|------|------|
| `protocol.json` | A(原基线 Hard NMS) vs B(Linear Soft-NMS) 实验协议 + 边界 |
| `model_manifest.json` | 三 fold checkpoint 路径/SHA256、config/测试标注 SHA256、git、环境 |
| `export_m0_candidates.py` | GPU：每图一次前向，导出候选缓存 + 同前向 Hard-NMS 参考 |
| `replay_nms.py` | 回放（CUDA `batched_nms`）+ 逐框验收 |
| `candidates/fold{6,7,8}.json` | 候选缓存（teacher2 ROI 回归后、NMS 前，含 p/box/proposal_id） |
| `acceptance/fold{6,7,8}/*.json` | Hard NMS 回放逐框验收结果 |
| `step4_acceptance.md` | 步骤 4 验收报告（含 CPU/CUDA 同分排序发现） |

## 关键约束（本步遵守）

- 未使用已经过 Hard NMS 的 `predictions.bbox.json` 作为 Soft-NMS 输入。
- 未提前按分数截 100 个候选；候选缓存为完整 NMS 前集合。
- 后处理回放未传入 GT 或真实 IoU。
- 回放算子（`batched_nms`）在 **CUDA** 上执行，与官方推理同设备（见 step4_acceptance.md）。
