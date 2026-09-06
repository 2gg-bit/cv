# M1：轻量 RoI 定位质量分支（探索性实验）

本实现是待验证的研究假设，不是已经证实有效的新方法，也不单凭添加质量头主张论文创新。
目标是检查“分类分数与定位质量排序不一致”是否能由辅助监督改善。
**本次交付不启动训练，不更改原 baseline 配置和已有实验产物。**

## 1. 实验边界

| 版本 | 质量头训练 | 推理排序 | 使用权重 |
| --- | --- | --- | --- |
| M0 | 无 | 原分类分数 | 已完成 baseline NMS |
| M1-A | 有 | 原分类分数 | 新 M1 最终 checkpoint |
| M1-B | 同 M1-A | 分类分数 × 预测定位质量 | 与 M1-A **同一** checkpoint |

M1-A 对比 M0 检查辅助训练的影响；M1-B 对比 M1-A 检查排序的影响。
M1-A/B 不需要分别训练。不得在测试集上搜索乘积指数、损失权重、阈值或最佳 checkpoint。
三个 fold 共享一个测试集，这一轮仍是探索性证据；论文结论需要后续独立验证、更多种子和消融。

- 沿用原 Phase1/2 权重、fold 6/7/8 的原 3 张标注图、batch=3、FP16、32000 iter 和原测试预处理。
- 新配置只改 RoI head 和输出目录，不更换 backbone/FPN，不引入额外标签或数据集。
- 质量头输入：原 bbox RoI 特征的空间平均池化 + **停止梯度**的 4 维回归量。
- 结构：`(256+4) → 64 → ReLU → 1`，每个分支增加 16,769 参数。
- 监督：仅 `sup1` / `sup2` 的正样本；目标为解码回归框与其分配到的真实 GT 的 IoU。
  目标和回归量停止梯度，对质量 logit 使用 BCEWithLogits（内部包含 sigmoid），质量损失权重固定 1.0。
  原 `sup2` 整体 0.2 权重照旧，所以它也作用于新增的 SAR 质量损失。
- 质量损失可更新共享 RoI 特征；这正是 M1-A 要单独测量的辅助训练效应。
- 不给伪标签添加质量监督，不改变双教师融合、共享 proposals、抖动不确定度和 EMA 日程。
  训练生成伪框继续走原 `simple_test_bboxes`，无论评估排序开关如何设置。
- M1-B 先按**原分类分数** `p_ship > score_thr` 保留候选，再用 `p_ship * sigmoid(q_logit)`
  做 NMS 排序及导出的 COCO score；**不对乘积再次套用 0.05 阈值**。
  NMS IoU、max_per_img、坐标和原图缩放方式保持不变。只支持当前单类 bbox 普通测试，不支持质量排序 TTA/ONNX。

## 2. 权重加载与开关

新配置：`configs/reproduce/phase3_dual_teacher_ssdd_m1.py`。

- `quality_enabled=False, quality_inference=False`：没有新增 state keys；严格加载原 baseline checkpoint，走父类原路径。
- `quality_enabled=True, quality_inference=False`：M1 训练及 M1-A 推理（默认）。
- `quality_enabled=True, quality_inference=True`：仅 M1-B 正式推理排序发生变化。

从旧 Phase1/2 初始化时，仅允许新增 `roi_head.quality_head.fc1/fc2` 的四个 weight/bias 键整体缺失。
同对 teacher/student 复制同一份新头初始化，其余所有旧参数及 buffer 继续校验名称、形状、有限值和复制后逐项相等。
部分质量参数缺失、旧参数缺失、未知键或尺寸错误均拒绝。最终复制仍使用 `strict=True`。
不同 Phase1/2 的校验排除新增质量头，防止随机新头掩盖误用同一份原检测器权重的问题。
恢复或评估完整 Phase3 checkpoint 仍要求完整四分支严格匹配，**不能**把 baseline Phase3 当作 M1 resume 权重。

## 3. 训练机验收顺序

以下命令在仓库根目录、原可用的 MMDetection/CUDA 环境中运行。先 `git pull --ff-only`。
所有输出路径必须是新的；不要覆盖已有 baseline 日志、JSON 或权重。

### 3.1 CPU 单元测试及真实初始化

```bash
python -m pytest -q tests/test_dual_teacher_baseline.py tests/test_quality_head.py \
  tests/test_quality_checkpoint.py tests/test_m1_config_and_routing.py tests/test_m1_export.py

python tools/check_dual_teacher_init.py configs/reproduce/phase3_dual_teacher_ssdd_m1.py \
  --cfg-options fold=6 percent=3
```

对 fold 7/8 重复初始化检查。必须看到仅四个质量头张量获准初始化，且最终
`T1=S1, T2=S2, T1!=T2; fusion=NMS, fusion_iou=0` 为 PASS。

### 3.2 关闭模块的真实预测等价性（先 fold 6）

对同一 baseline checkpoint、同一软件/GPU 环境，分别用原配置和关闭模块的新配置重新导出。

```bash
python tools/eval_teacher2_export.py configs/reproduce/phase3_dual_teacher_ssdd.py \
  work_dirs/phase3_dual_teacher_baseline_nms/3/6/iter_32000.pth \
  --fold 6 --out-dir eval_export/m1_acceptance_reference/fold6

python tools/eval_teacher2_export.py configs/reproduce/phase3_dual_teacher_ssdd_m1.py \
  work_dirs/phase3_dual_teacher_baseline_nms/3/6/iter_32000.pth \
  --fold 6 --out-dir eval_export/m1_acceptance_disabled/fold6 \
  --cfg-options model.roi_head.quality_enabled=False model.roi_head.quality_inference=False

python tools/compare_prediction_exports.py \
  eval_export/m1_acceptance_reference/fold6 eval_export/m1_acceptance_disabled/fold6
```

要求图像覆盖、类别和逐框坐标/分数一致，而不仅是三位小数 AP 一致。
原配置应复现 fold 6 AP=0.459。失败先排查环境/权重/代码路径，不调整阈值凑数；再对 fold 7/8 检查。

### 3.3 100 iter 功能冒烟（不是正式结果）

这一步会在训练机实际进行 100 步训练，需由操作者确认前两步通过后启动。
不做测试集评估；冒烟目录和正式实验完全隔离。沿用已记录的种子（以下为 678）。

```bash
python -m torch.distributed.launch --nproc_per_node=1 \
  tools/train.py configs/reproduce/phase3_dual_teacher_ssdd_m1.py \
  --launcher pytorch --seed 678 --no-validate \
  --work-dir work_dirs/m1_smoke/3/6 \
  --cfg-options fold=6 percent=3 runner.max_iters=100 checkpoint_config.interval=100 log_config.interval=10
```

检查初始化 PASS、日志中有有限的 `sup1_loss_quality` / `sup2_loss_quality`，没有
`unsup1_loss_quality` / `unsup2_loss_quality`；检查无持续 AMP overflow、NaN、OOM 或 DDP unused-parameter 错误。
记录单步时间和显存峰值，确认 16GB 卡留有余量。CPU 单测不代替这一步。
冒烟 checkpoint **不用于**正式训练的 resume，也不能据其测试 AP 决定超参数。

### 3.4 正式实验（验收通过后再决定启动）

首先只做 fold 6 的一次完整、预先固定的实验；成功后按相同协议补 fold 7/8，避免同时跑多组试错。

```bash
python -m torch.distributed.launch --nproc_per_node=1 \
  tools/train.py configs/reproduce/phase3_dual_teacher_ssdd_m1.py \
  --launcher pytorch --seed 678 --no-validate --cfg-options fold=6 percent=3
```

输出为 `work_dirs/phase3_dual_teacher_m1_quality/3/6`，从原 Phase1/2 新开始，不加 `--resume-from`。
这里 `--no-validate` 只关闭中途测试，最终固定取 `iter_32000.pth`；不改变训练迭代或优化设置。

### 3.5 同一个 M1 checkpoint 导出 A/B

```bash
python tools/eval_teacher2_export.py configs/reproduce/phase3_dual_teacher_ssdd_m1.py \
  work_dirs/phase3_dual_teacher_m1_quality/3/6/iter_32000.pth \
  --fold 6 --out-dir eval_export/m1_a/fold6 \
  --cfg-options model.roi_head.quality_inference=False

python tools/eval_teacher2_export.py configs/reproduce/phase3_dual_teacher_ssdd_m1.py \
  work_dirs/phase3_dual_teacher_m1_quality/3/6/iter_32000.pth \
  --fold 6 --out-dir eval_export/m1_b/fold6 \
  --cfg-options model.roi_head.quality_inference=True
```

A/B 的预测本来可能不同，不用“逐框必须相同”的比较器验收它们。
报告 M0/M1-A/M1-B 的 AP、AP50、AP75、APs/AR、同 GT 的定位排序损失以及每图配对差异。
提高 AP75 才与当前定位排序假设直接相关，但主指标仍须预先指定并完整报告，不能只挑上涨指标。
不把 NMS 移除框数或 oracle 覆盖率直接换算成 AP 增益。

## 4. 验证范围

本地 CPU 测试验证模块数学、监督路由、开关行为、受控初始化、EMA 参数覆盖和导出校验。
本地并无训练服务器的 CUDA/MMCV 算子及真实权重，不能据此声称已验证全图数值一致、显存或检测精度提升。
上述真实模型初始化、全 232 图等价检查和 100 iter 冒烟，是正式长训前必须补做的门槛。
