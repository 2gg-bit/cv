# Sup2 定位约束与工程修复（2026-10-01）

本次是代码交付，不启动训练。已完成的小背景重加权三折保留为探索性结果，
不因工程修复而重跑。旧源码以 `6485dec` / `0eed22a` 保留，旧 source receipt、
验收 JSON、权重与数值结果不改写。

## 实验定义

新组为 **B0 + sup2 GIoU**，默认实验系数 beta=1.0，在结果产生前固定。
模块本身默认关闭；关闭或 beta=0 直接返回父类原路径，不增加损失键。
不同时开启小背景重加权、M2、FG、PG 或其他模块。

- 继承 StandardRoIHead，只在 `img_metas.tag == 'sup2'` 的正 RoI 上增加 `loss_giou`。
- 原 RPN、分类和 bbox L1 损失、assignment/sampling、回归目标及推理保持原实现。
- 用原 bbox coder 解码预测 delta，再与该正 RoI 对应的 pos_gt_bboxes 计算 GIoU。
  不能在编码后的 delta 上计算 IoU。class-specific head 按正例类别取对应四维。
- 使用增强后同坐标系的预测框/真实框，在 FP32 中计算几何项。
- `loss_giou = beta * sum(1-GIoU) / N`；N 为 sup2 图上采样的全部 RoI 数，
  与原 RoI 回归损失的计数基准一致，不是只除以正例数。
- DualTeacher 原有的外层 sup2_weight=0.2 继续作用于它，正式日志键为
  `sup2_loss_giou`。sup1 和两路无监督分支不增加此项。
- 无正例时返回连接到 bbox_pred 的有限零损失，不伪造“命中”验收。
- 不新增参数或 buffer；原 B0、Phase1/Phase2 的 state_dict 可严格加载。

GIoU 是已有方法，本组隔离检验它在该框架的作用，不单凭接入它主张新方法。
参考：[GIoU，CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Rezatofighi_Generalized_Intersection_Over_Union_A_Metric_and_a_Loss_for_CVPR_2019_paper.html)。

## 从每折实际 B0 生成配置

用服务器已冻结的各折 B0。生成器保留其数据、增强、Phase1/Phase2、优化器、EMA、
32000 步、混精和推理设置。新模板 phase3_dual_teacher_ssdd_sup2_giou.py 只作参考。

```bash
cd /home/xcc/dual_teacher_project/DualTeacher_ablation
conda activate dt
python tools/train_ablation.py --check-source-only
set -e
for fold in 6 7 8; do
  python tools/prepare_ablation_suite.py "ablation_configs/fold${fold}_seed678/b0.py" \
    --out-dir "ablation_configs/giou_v1/fold${fold}_seed678" \
    --work-root "work_dirs/giou_v1/fold${fold}_seed678" \
    --seed 678 --experiments b0 sup2_giou --giou-weight 1.0
done
```

输出目录和训练 work-root 必须全新。生成 b0.py 只是保留对照配置，**不要求重训 B0**。
核对每折 manifest 中自己的 Phase2 与标注哈希。新组从 Phase1 / 对应折 Phase2 初始化，
不从任何最终 Phase3 权重续训。源码收据现在包含 11 个模块，哈希变化来自本次提交。

小背景配方也可重建：同一命令改用 `--experiments b0 reweight_l1`，选择另一套全新路径。
另有 phase3_dual_teacher_ssdd_reweight_l1.py 模板。这补足可生成配方的能力，
**不宣称生成文件与缺失的历史三份 reweight_l1.py 逐字节相同**。历史配置哈希继续保留；
需要字节级复核时取服务器原文件。本次不安排重加权重训或组合实验。

## 训练机验收

每折分别执行。任一 failed 先停下修复，不得循环跳过失败直到找到通过。

```bash
fold=6
SUITE="ablation_configs/giou_v1/fold${fold}_seed678"
python tools/train_ablation.py --check-init "$SUITE/sup2_giou.py"
python tools/train_ablation.py --check-step "$SUITE/sup2_giou.py" \
  --seed 678 --batch-index 0 --out-dir "$SUITE/giou_acceptance_batch0"
```

新检查执行 off / off_repeat / on / beta=0：CPU/CUDA/Python/NumPy RNG 全部恢复；
同变体前向逐位复现，原损失逐位不变，唯一新增项为 sup2_loss_giou。
同时检查辅助损失能反传到 student2.fc_reg，teacher 无梯度、参数/buffer 未变。
混精跟随模型 auto_fp16 装饰器路径，用配置对应的新建 GradScaler 做 scale-backward-unscale；
不执行 optimizer.step、scaler.step/update 或 EMA。不用 AMP 反向逐张量差异归因模块效果。

无有效正例/辅助损失为零标 incomplete（exit 2），可换新输出目录检查另一批；
与 failed（exit 1）区分。CPU 单测不能代替这里的真实 CUDA 验收。

## 后续训练、评估（训练机执行，本次未启动）

三折固定配方，不依中途增益正负取消后续折。使用 `--seed 678 --no-validate`，
不加 `--deterministic`。先单卡顺序完成训练，GPU 空闲后串行评估。
以下是一折示例，须先通过 init/step 验收；fold7/8 同样执行。

```bash
fold=6
SUITE="ablation_configs/giou_v1/fold${fold}_seed678"
WORK="work_dirs/giou_v1/fold${fold}_seed678/sup2_giou"
test ! -e "$SUITE/giou_train.log" || exit 1
test ! -e "$SUITE/giou_train_exit_code.txt" || exit 1
set -o noclobber
set +e
python -m torch.distributed.launch --nproc_per_node=1 tools/train_ablation.py \
  "$SUITE/sup2_giou.py" --launcher pytorch --seed 678 --no-validate \
  > "$SUITE/giou_train.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$SUITE/giou_train_exit_code.txt"
test "$rc" -eq 0 || exit "$rc"
# 核对满 32000 步、三类异常、最终权重哈希和来源记录后再评估。
python tools/train_ablation.py --eval "$SUITE/sup2_giou.py" \
  "$WORK/iter_32000.pth" --fold "$fold" --out-dir "$SUITE/sup2_giou_evaluation"
```

三折评估成功后，以各折原新 B0 为对照，CPU 复算未舍入配对增量：

```bash
python tools/recompute_paired_coco_delta.py --expect-pairs 3 --require-checkpoints \
  --pair fold6 ablation_configs/fold6_seed678/b0_evaluation ablation_configs/giou_v1/fold6_seed678/sup2_giou_evaluation \
  --pair fold7 ablation_configs/fold7_seed678/b0_evaluation ablation_configs/giou_v1/fold7_seed678/sup2_giou_evaluation \
  --pair fold8 ablation_configs/fold8_seed678/b0_evaluation ablation_configs/giou_v1/fold8_seed678/sup2_giou_evaluation \
  --out ablation_configs/giou_v1/paired_coco_delta.json
```

主指标 mAP，并列 AP75/APs/AR@100。仍用已参与选型的 232 图，定位为探索性比较。
当前没有 GIoU 精度结果。现有 B0 是固定参考，本次没有声称训练随机性消失。

## 工程修复与历史证据

1. 小背景详细诊断默认关闭；验收主动启用最多 16 条记录。训练不转换/累积完整 ROI 列表，权重公式不变。
2. 复算最终判定涵盖两侧全部 checks。存在的 checkpoint 哈希不符必须失败；
   --require-checkpoints 要求权重存在并核验。不传时缺失记 unknown，不能称作权重核验通过。
   相同权重哈希、重复 pair/label、缺折、产物冲突均不能产生完整成功的三折报告。
3. mmdet 源码默认在仓库同级 thirdparty，可用 --vendored-coco-py 指定实际文件。
4. recipe_freeze.json 仅纠正 sup2 的文字解释；旧数值、实现哈希和覆盖限制保留。

旧重加权护栏复核只需对原预测目录运行修复后的 CPU 复算，写入新文件名，例如
paired_coco_delta_engineering_recheck.json；不重新导出预测、不覆盖历史报告。
开发机没有服务器原预测和权重，不能宣称该实算已经完成。

最初 verify_small_bkg_reweight.py 的“以重复差当容差”结论已被 supplement /
amp_scaled_backward 记录取代；最初产物保留历史，不能单独当最终验收。

本次本地验证使用 Python 3.6 / Torch 1.7.0 CPU 与 Torch 2.5.1 CPU 两套环境。
测试执行实际类/方法代码，外部 MMDetection 接口用小型 fixture 替代；没有 CUDA RoI 算子或真实训练输入。
额外与 Git 中 6485dec 的旧重加权实现对照 40 个 CPU 案例，目标、分类损失、梯度和 RNG 逐位一致。
