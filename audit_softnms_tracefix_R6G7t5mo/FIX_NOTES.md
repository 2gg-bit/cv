# 追踪记录索引错配修复说明

## 具体代码原因

旧版 `eval_ab.py` 的 `replay_hard_nms`（A 组 Hard NMS）中：

```python
dets, _ = batched_nms(boxes_c, p_c, labels, dict(type='nms', iou_threshold=0.5))  # 丢弃了 keep
dets = dets[:100].cpu()
vidx = valid.nonzero(as_tuple=False).squeeze(1)   # 过滤后索引 -> 原始索引
for k, d in enumerate(dets):
    orig_idx = int(vidx[k].item())                 # 错误：把输出排名 k 当成了过滤后索引
```

`batched_nms` 返回 `(dets, keep)`，其中 `dets` 已按分数降序排列，`keep[k]` 才是第 k 个输出框在
**过滤后数组**中的下标。旧代码丢弃 `keep`，用输出排名 `k` 直接索引 `vidx`（过滤后→原始映射），
等于“把输出排名当 proposal_id 索引”。当 NMS 输出顺序与输入顺序不一致（几乎总是如此，因为按分数排序），
追踪记录里的 `proposal_id`/`original_p` 就会指向错误的候选。

- 影响范围：**仅追踪记录（A_changes.json 的 proposal_id/original_p）**，不影响预测框与分数
  （预测直接取自 `dets`，与官方预测逐框一致，已复核）。
- B 组（soft_nms）无此问题：`soft_nms` 返回的 `inds` 被正确使用（`orig_idx = vidx[inds[k]]`）。

## 正确关系（已修复）

```
原始候选及ID
  -> 同步执行分数过滤 (valid)
  -> batched_nms 返回 keep          # keep[k] = 过滤后数组下标
  -> 原始下标 = vidx[keep[k]]
  -> 同步排序、截断 框/ID/原分数/输出分数
  -> 生成追踪记录
```

修复后：`orig_idx = int(vidx[int(keep[k].item())].item())`。原分数始终从真实候选缓存
`cands[orig_idx]['p']` 读取，未用 `output_score` 回填。

## 验证结果（三组全通过）

| 检查 | A | B |
|------|---|---|
| 回放预测 vs 已验证预测逐框一致 | ✅ 位级一致 | ✅ 位级一致 |
| 追踪记录可回查缓存（ID/原分数/原框） | ✅ 全通过 | ✅ 全通过 |
| 分数规则（A: output==original；B: output≤original） | ✅ 全通过 | ✅ 全通过 |
| 输出对应（每条与最终预测 rank/框/分数一致） | ✅ 全通过 | ✅ 全通过 |
| 完整性（追踪条数==输出框数） | ✅ | ✅ |
| 原预测/指标/缓存 SHA256 未变 | ✅ 16 个文件字节一致 | ✅ |

几何统计 `max_iou_lt_01` 与既定对账值一致（第一组 17589/22329、第二组 21598/26219、第三组 17165/22126）。
