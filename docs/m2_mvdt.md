# M2 + MVDT：动态伪标签分类准入

这是一个默认关闭的训练改动，借鉴 *Dual-Domain Teacher for Unsupervised
Domain Adaptation Detection* 的 minimum variance-based dynamic threshold
（MVDT，公式 2-4）。保留现有 M2 正 RoI 分类权重，在它之前增加动态准入。
尚未得到本配置的完整训练精度结果，不宣称 AP 已提升。

## 改了什么

- 对两教师融合后的候选检测分数统计一次，不重复计入两个教师的同一融合集合。
- 将分数降序排列，在两组各至少两个样本、边界分数不同的切分中，精确最小化
  两组内部平方误差总和。用 float64 累计矩计算，不使用直方图近似或随机抽样。
- 第 1-1000 步保持原来的 `score > 0.9`。第 1000 步结束后首次更新，之后每
  1000 步更新；新阈值从下一步开始对两个学生同时生效。
- 更新成功后阈值取高分组最低分 `s_m`，使用 `score >= s_m`，避免丢掉切分边界。
  同分候选不会被分到两侧。样本少于 128、或没有满足条件的分数边界时沿用前值。
- 无标签候选分数来自训练图像；不读取开发集标签、AP 或测试集。
- 使用固定迭代窗口是对本项目 IterBasedRunner 的适配，原论文按 epoch 更新。

M2 继续对同一采样 RoI 的双教师概率使用 `min(p_T1, p_T2)`，保持原 `Z0` 计算
规则、背景概率权重和损失系数。动态准入会改变采样内容，所以 `Z0` 的具体数值
不要求与旧运行相同。初筛 0.5、RPN 阈值 0.9、回归抖动阈值 0.02、框融合、
推理和后处理未改。没有启用 M1/M3、Mamba 骨干或前景辅助头。

## 文件

- `ssod/models/mvdt.py`：精确切分、分布式分数汇总、阈值状态与恢复。
- `ssod/models/dual_teacher.py`：两个无监督 RoI 分类入口；每个 forward 结束后更新一次。
- `configs/reproduce/phase3_dual_teacher_ssdd_dev_m2_mvdt.py`：独立实验配置。
- `tools/train_m2_mvdt.py`：在实际 worker 中固定代码根并记录加载来源。

默认 `max_scores=1000000`，分数缓存约 4 MB/模型。超过容量会明确报错，不静默
截断或偏向高分采样。若训练设置确实需要更多候选，可在开始新运行前增大容量。
每个 rank 每步汇总一次候选，所有 rank 持有相同的完整窗口；单卡没有通信开销。
FP16 转换不会降低阈值、累计分数和计数的精度。控制器没有可训练参数，不参与 EMA。

阈值、步数、更新次数、未完成窗口及配置均进入 checkpoint。因此从窗口中途恢复也
能继续原统计；配置不一致或缺少状态时，即使非严格加载也报错，避免静默重置。
推理时无需创建该控制器，训练状态不影响 teacher2 的检测结果。

## 在训练电脑启动

在包含数据路径、`ssdd_dev_protocol/data` 和原 Phase1/2 权重的仓库根目录执行。
下例沿用 seed 678、3 张标签、fold 6，另用新目录，保留已有结果。

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# 只检查实际导入的源文件，不启动训练；需在原 MMDetection 环境执行。
python tools/train_m2_mvdt.py --check-source-only

# 检查真实 Phase1/2 权重加载；不会启动训练。
python tools/check_dual_teacher_init.py \
  configs/reproduce/phase3_dual_teacher_ssdd_dev_m2_mvdt.py \
  --cfg-options fold=6 percent=3

mkdir -p work_dirs/dev_ssdd/phase3_dual_teacher_m2_mvdt_seed678/3/6
set -o pipefail
python -m torch.distributed.launch --nproc_per_node=1 \
  tools/train_m2_mvdt.py \
  configs/reproduce/phase3_dual_teacher_ssdd_dev_m2_mvdt.py \
  --launcher pytorch --seed 678 --no-validate \
  --work-dir work_dirs/dev_ssdd/phase3_dual_teacher_m2_mvdt_seed678/3/6 \
  --cfg-options fold=6 percent=3 2>&1 | \
  tee work_dirs/dev_ssdd/phase3_dual_teacher_m2_mvdt_seed678/3/6/launcher.log
```

`tools/train_m2_mvdt.py` 在每个 worker 导入 `ssod` 之前将本仓库置于 `sys.path`
和 `PYTHONPATH` 最前方，然后核对实际加载路径，并输出 `[MVDT source]` JSON，
包含 `ssod`、`dual_teacher.py`、`mvdt.py` 的路径和 SHA256。若仍加载其他目录，
立即报错。`launcher.log` 保存这些启动信息，解决 editable install 指向另一份
DualTeacher 时启动器与 worker 加载不同实现的问题。

第 1000 步附近应出现：

```text
[MVDT] step=1000 samples=... threshold=... updated=True; effective from next training forward
```

若候选太少或全部同分，`updated=False` 是保留旧阈值；结合 `samples` 检查实际
分数覆盖。修正版在第 1 步及每 50 步直接写入 INFO 文件日志，例如：

```text
[MVDT threshold] step=1000 mvdt_cls_threshold=0.900000 next_cls_threshold=0.812345
```

`mvdt_cls_threshold` 表示这一训练步两个学生实际使用的值，`next_cls_threshold`
表示本步结束后的值。在更新边界二者可能不同。此日志不经过 DEBUG 默认级别的
`log_every_n`，也不会被已启用的 wandb 分流。M2 原有正/负 RoI 统计继续保留。
初始提交 `ffa5704` 只有每次更新的 `[MVDT]` INFO 行，周期阈值字典被 DEBUG
级别过滤；这是日志缺失，不影响当时的阈值更新或 checkpoint 状态。

## 恢复与评估

新实验从原 Phase1/2 初始化。不要把旧 M2/M3 完整 checkpoint 当作带 MVDT 历史
的运行恢复；使用本配置产生的完整 checkpoint 和相同 MVDT 参数显式恢复：

```bash
python -m torch.distributed.launch --nproc_per_node=1 \
  tools/train_m2_mvdt.py \
  configs/reproduce/phase3_dual_teacher_ssdd_dev_m2_mvdt.py \
  --launcher pytorch --seed 678 --no-validate \
  --work-dir work_dirs/dev_ssdd/phase3_dual_teacher_m2_mvdt_seed678/3/6 \
  --resume-from work_dirs/dev_ssdd/phase3_dual_teacher_m2_mvdt_seed678/3/6/iter_4000.pth \
  --cfg-options fold=6 percent=3
```

继续用相同开发集、teacher2 和既有后处理设置与 M2 比较，不改数据划分。优先看
AP、AP75、APs、AR@100 以及入选伪框数；本改动只改变分类准入，不保证定位或
召回一定同时改善。原跨种子编排器的人工放行、账本和哈希约定不由本代码更改，
不要将新配置作为旧 run 的自动续跑项。

## 与冻结跨种子队列并存

若 `DualTeacher` 正在运行 `ffa5704`，本次兼容/日志修补不要求重启训练。保留
当前运行目录和源码版本，修补可在另一个 checkout 验证，或等本轮结束后拉取。
修改磁盘上的源码不会让已导入的 Python 函数自动升级，反而会使后续磁盘哈希与
启动记录不一致。当前运行的每千步 `[MVDT]` INFO 更新行与 checkpoint 状态仍有效。

另一个冻结目录 `DualTeacher_m3` 不应跟随此仓库更新模型源码。两目录源码已经
不同，`cwd=DualTeacher_m3` 不能保证实际 worker 使用它：editable install 仍可能
将 `ssod` 指向 `DualTeacher`。队列重启应使用已单独交付的 v4.1 启动/核验流程：

1. 核实并停止旧的交接等待编排器，保留其输出和 run #1 产物；确认停止的不是
   当前 M2+MVDT worker。旧 v3 尚能启动 run #2 时，不创建 `HANDOVER_RELEASED`。
2. 在冻结目录安装 v4.1 新增的控制工具，保持模型、配置、权重不变。实际子进程
   显式设置 `PYTHONPATH`，使用 `crossseed_train_worker_v4_1.py`。仅在核验器里
   固定路径，或只执行启动器中的 `import ssod`，都不足以解决问题。
3. 先运行 v4.1 的实际 launcher/worker 预检及 dry-run。worker 必须核对实际
   导入路径、注册的 DualTeacher 类、预期源码哈希，并写绑定启动 manifest 的
   回执。预检须标明 `training_started=false`。
4. 用 `--handover-only` 单独处理 run #1，避免放行后自动开启后续训练。run #1
   的历史运行时来源证据缺口仍须保留，不能用这次预检补写为旧运行的启动证据。
   之后剩余队列必须继续使用同一套带 worker 回执的 v4.1 流程。

以上是另一台训练电脑上的迁移要求；本提交没有远程部署队列，也没有创建放行文件。

## 检查范围

```bash
python -m pytest -q tests/test_mvdt.py tests/test_mvdt_integration.py tests/test_mvdt_launcher.py
python -m pytest -q tests
```

新增测试覆盖：切分结果与直接枚举对照、同分/空集合、更新时点、两个学生实际
分类方法、融合候选只计一次、窗口中途恢复、损坏状态拒绝、FP16 状态保持、
默认关闭时兼容旧模型，以及两个真实 Gloo 进程中的不等长/空批次汇总。
完整检测器构建使用小型 CPU fixtures；这些检查不替代训练机上的 CUDA 算子、
真实数据前向和完整训练精度验证。

2026-09-21 初始提交的本地检查是 Python 3.12 / PyTorch 2.5.1 CPU 下
259 通过、2 跳过，并非训练电脑的 dt 环境结果。随后训练电脑报告两个
PyTorch 1.7 浮点 `.any()` 不兼容：新 MVDT 测试的缓存判断，以及已有 M3
检查工具的负样本权重判断。本修补均显式转为 bool 后调用 `.any()`。

修补后的本地检查：

- Python 3.12 / PyTorch 2.5.1 CPU：完整测试集 260 通过、85 个子测试通过；
  2 项分别因缺少 CUDA 和 MMDetection 环境跳过。
- Python 3.6.4 / PyTorch 1.7.0 CPU：`test_mvdt.py`、`test_mvdt_integration.py`、
  `test_mvdt_launcher.py`、`test_m3_roi_head.py` 共 54 项通过，覆盖两处兼容
  修复、阈值/恢复/日志、实际分类入口及 M3 检查观察器。
- 两个版本的上述分布式 Gloo 测试均实际执行并通过。

这仍不是训练电脑原 CUDA + MMCV 1.3.9 + MMDetection 2.16 的完整检测器验证。
