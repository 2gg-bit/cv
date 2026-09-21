# M2 + 前景位置辅助监督

本实验在 M2 上增加真实框监督的 P2 位置热图辅助头，检验额外位置监督能否改善
船与背景的特征区分。它受 FGBG-Net 前景定位思路启发，是简化迁移设计，不是
完整 FGBG-Net 复现，也没有已验证的 AP 增益。

参考：Ma 等，*Dense-Weak Ship Detection Based on Foreground-Guided Background
Generation Network in SAR Images*，TGRS 2025，DOI 10.1109/TGRS.2025.3572095；
[作者仓库](https://github.com/Xidian-AIGroup190726/FBGBNet)。

## 实现与实验边界

- 独立配置：`configs/reproduce/phase3_dual_teacher_ssdd_dev_m2_fg.py`。
- 注册 `ForegroundRoIHead`，复用检测器已提取的首层 FPN（当前是 stride=4 的 P2），
  经过 256→32 的 3×3 卷积、ReLU、32→1 的 1×1 卷积，预测位置热图。
  每个分支新增 73,793 个参数；四分支均保存此头，EMA 按原规则更新教师。
- 只有 `sup1`（带框光学）和 `sup2`（带框 SAR）显式启用辅助损失。
  无标签图、伪标签分类/回归、教师伪框生成都不运行辅助头。
- 使用增强后的真实框坐标生成高斯位置目标；中心按 stride 向下取整，宽高
  分别确定 sigma=框尺寸/(6×stride)，sigma 至少为半个网格，绘制到 3 sigma。
  多个目标取逐点最大值，小框至少有一个值为 1 的中心。同一网格重合的中心
  合并为一个峰，因此这是粗位置监督，不是实例分割或精确船体掩码。
- 忽略图像填充与 `gt_bboxes_ignore` 区域（ignore 优先）；真实空标注图贡献背景
  损失。仅完整标注图适用，不能把无标签图的无框区域当成已知背景。
- 使用 Gaussian focal loss，alpha=2、beta=4；每图按有效中心数归一化（至少 1），
  再平均图像。卷积、目标与损失为 FP32，梯度仍回传共享 FPN/backbone。
  输出偏置初始化为 log(0.01/0.99)，减少初始海量背景的损失。
- 初始实验固定 `foreground_loss_weight=0.1`。这是待验证超参数，不是从 dev AP
  拟合的最佳值。原 `sup2` 整体 0.2 权重仍作用于其辅助损失，因此总损失增加
  `0.1×L_fg_sup1 + 0.02×L_fg_sup2`。每 50 步的常规日志包含
  `sup1_loss_foreground`、`sup2_loss_foreground`（已乘这些系数）。
- M2 正 RoI 加权及 Z0 规则、背景权重、RPN、回归、融合、0.9 分类准入保持原设置。
  本配置显式关闭 MVDT/M3；辅助头默认关闭，旧 M2 配置仍使用 StandardRoIHead。
- 推理完全继承原 RoI 方法，不计算热图，不改变分数或 NMS。辅助参数保存在权重中，
  不表示推理会执行它们。首轮不增加特征门控，不使用开发集标签训练。

## 初始化与恢复

从原 Phase1/2 权重重新启动 Phase3，不必重训 Phase1/2。加载器只允许四个明确的
`roi_head.foreground_head.{conv,out}.{weight,bias}` 新参数共同缺失；原检测参数
仍逐项核验。新增头先从对应教师取初值，再复制给学生，确保 T1=S1、T2=S2。
判断 T1≠T2 时排除辅助头，避免随机辅助头掩盖错用相同 Phase 权重的问题。

完整 FG checkpoint 必须配合相同结构和实验配置恢复；不能用旧 M2 完整权重
`--resume-from` 来冒充新实验。缺少/多出辅助头或头尺寸不符时，即使 MMCV 使用
非严格加载也报错。恢复时保持损失权重、数据、seed 等实验设置一致。
评估也使用 FG 配置构造模型，完整加载四分支后选 teacher2；不需要 Phase1/2 文件。

## 在训练电脑执行

在含数据、`thirdparty/mmdetection` 和 Phase1/2 权重的独立 checkout 根目录执行。
使用原环境 torch 1.7.0 / mmcv 1.3.9 / mmdet 2.16.0。
以下命令固定 seed678 / 3-shot / fold6，独立输出，不覆盖已有实验。

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CFG=configs/reproduce/phase3_dual_teacher_ssdd_dev_m2_fg.py

# 实际加载的六个模块必须来自当前 checkout，输出路径与 SHA256。
python tools/train_m2_fg.py --check-source-only

# CPU 检查真实 Phase1/2 权重，严格核验四分支；不训练。
python tools/train_m2_fg.py --check-init "$CFG" \
  --cfg-options fold=6 percent=3

# 一批真实训练数据的 CUDA/FP16 前后向；输出目录必须不存在。
# 检查辅助头 off/on 时原损失相同、仅 sup1/sup2 增加辅助损失、
# 两个学生辅助头梯度非零、所有梯度有限、教师无梯度。不更新参数。
mkdir -p work_dirs/fg_acceptance
set -o pipefail
python tools/train_m2_fg.py --check-step "$CFG" \
  --seed 678 --out-dir work_dirs/fg_acceptance/seed678_fold6 \
  --cfg-options fold=6 percent=3 \
  2>&1 | tee work_dirs/fg_acceptance/seed678_fold6_launcher.log
```

上述检查通过后再开始正式训练。新运行目录若已存在，先核对用途，不能直接覆盖。

```bash
(
RUN=work_dirs/dev_ssdd/phase3_dual_teacher_m2_fg_seed678/3/6
mkdir -p "$(dirname "$RUN")" || exit 1
mkdir "$RUN" || exit 1  # 已存在时停止，另取新目录或明确恢复
set +e
set -o pipefail
python -m torch.distributed.launch --nproc_per_node=1 \
  tools/train_m2_fg.py "$CFG" \
  --launcher pytorch --seed 678 --no-validate --work-dir "$RUN" \
  --cfg-options fold=6 percent=3 2>&1 | tee "$RUN/launcher.log"
train_status=${PIPESTATUS[0]}
printf '%s\n' "$train_status" > "$RUN/train_exit_code.txt"
exit "$train_status"
)
```

入口在每个 worker 导入 `ssod` 前置顶本仓库的 `sys.path` 与 `PYTHONPATH`，
核验实际加载路径；`[Foreground source]` JSON 保存在 `launcher.log`。
不要绕过入口用另一份 editable install 的 `tools/train.py` 启动。
保持当前正在训练或冻结交接的 checkout 不动；本实验不会创建任何人工放行文件。

## 评估与保留标准

训练 32000 步后，沿用同一 186 图 dev.json、teacher2、score_thr=0.05、
max_per_img=100 和原 Hard/Soft-NMS 参数。比较 M2 与 M2+FG：AP、AP50、AP75、
APs、AR@100，以及固定口径 TP/背景误检、score≥0.9 的背景误检。
主对照为 fold6 M2 Soft AP=0.5412484、Hard AP=0.5358486。
先判断这一单项实验是否提高精度，再决定是否扩到 fold7/8。不叠加 MVDT。

## 已完成的本地验证

Windows CPU，Python 3.12 / torch 2.5.1：全套测试 287 passed、2 skipped、
85 subtests passed。Python 3.6 / torch 1.7.0 CPU：前景模块、质量头及严格加载相关
测试 84 passed、1 skipped。跳过项是 CUDA 条件测试；不代表 GPU 已通过。

覆盖小框/边界/重叠目标、padding/ignore、空图、极端 logits、FP16 输入与参数的
FP32 计算、共享特征梯度、真实监督路由、原 0.2 分支权重、严格初始化、完整恢复、
教师 EMA 和实际导入来源校验。框架边界使用小型 fixture 执行实际类/方法。
本地没有 MMDetection 编译算子、训练图片与 Phase1/2 权重；真实整网 CUDA 验收和
AP 提升尚未执行，须在训练电脑运行上面的命令，不能将 CPU 测试当成训练结果。
