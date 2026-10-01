# 步骤 4 验收：完整候选缓存 + 原 Hard NMS 回放

## 结论

| fold | 复算框数 | 参考框数 | 图像覆盖 | 每图框数 | 逐框一致 | 官方 predictions 逐框一致 |
|------|---------|---------|---------|---------|---------|--------------------------|
| 6 | 2879 | 2879 | ✅ | 0 差异 | ✅ 位级一致 | ✅ 位级一致 |
| 7 | 3302 | 3302 | ✅ | 0 差异 | ✅ 位级一致 | ✅ 位级一致 |
| 8 | 2798 | 2798 | ✅ | 0 差异 | ✅ 位级一致 | ✅ 位级一致 |

**通过条件（逐框一致，非仅 AP 三位小数）已满足**：回放预测与同前向参考、以及与官方
`work_dirs/m0_seed678/evaluation/{fold}/predictions.bbox.json` 均逐框位级一致
（image_id、坐标、分数、类别完全相同）。

## 候选缓存

- 来源：`export_m0_candidates.py`，每个图像**仅一次** `_bbox_forward`，从同一份
  `cls_score`/`bbox_pred` 同时导出：
  - `candidates`：ROI 回归后、最终 NMS 前的完整候选（解码框，resize 坐标，含
    `proposal_id`、`box`、`p`=船类前景分数 `softmax[:,0]`）；
  - `reference_hard_nms`：同一前向的官方 Hard NMS 输出（原图坐标）。
- 完整性：每图候选数 = RPN 提案数（≤1000），未截断、未按分数预筛。
- 坐标/分数精度：float32 位级无损往返（JSON 往返 bit-exact）。
- 未使用任何 GT / 真实 IoU；仅前向冻结的 teacher2。

## 关键发现：算子设备导致精确同分候选的 NMS 排序歧义

初版回放出现 3/2879 框坐标不一致（分数完全相同）。排查定位为**两个候选的分类分数
位级相同**（例如 image 1071 的 proposal 1 与 proposal 266，`p=0x3f6cb3df`），
NMS 的 `scores.sort(0, descending=True)` 对完全相等的键**不稳定**，且
**PyTorch CPU sort 与 CUDA sort 对同分键的排序不同**：

- CPU `batched_nms` 保留 proposal 1（box `[170.11,245.54,…]`）；
- CUDA `batched_nms` 保留 proposal 266（box `[170.35,245.29,…]`）= 官方结果。

修复：回放中的 `batched_nms` 改为在 **CUDA** 上执行（与参考推理同一设备），
3 处同分歧义随之消失，三 fold 全部逐框位级一致。

## 结论口径

- 候选缓存完整、位级无损、与官方前向一致；回放逻辑正确。
- “逐框一致”的障碍不是候选遗漏/精度/坐标变换/顺序，而是**算子设备**
  （CPU vs CUDA 排序对同分键的稳定性的差异），已按官方设备修复，未放宽任何误差。
