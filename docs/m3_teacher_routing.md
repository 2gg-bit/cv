# M3 v1：分类可靠性与回归教师选择分开处理

这是待验证的框架实验，不是已经有效的方法。先完成下面的离线审计和 GPU 验收，
再决定是否正式训练。新增代码不会自动启动训练，也不修改、删除既有实验产物。

本次开发机验证：临时 Python 3.12 / PyTorch 2.14 CPU 环境中，`pytest -q tests`
为 **223 passed、2 skipped、85 subtests passed**。跳过项分别需要 CUDA 与真实 MMDetection。
另通过 Python 3.6 语法检查；这不等于已在训练机的 PyTorch 1.7/MMCV 1.3.9 上运行通过，
真实权重、实际数据管线与 GPU 前向/反向仍由以下门槛验证。

## 1. 修改了什么

- 保留 M2：仅在无监督 RoI 分类的正样本上，使用
  `a_i * stopgrad(min(p_T1, p_T2))`，仍除以修改权重前的 `max(Z_0, 1)`。
- M3 复用原有 10 次 jitter 前向，另外保留每位教师对**同一融合候选**输出的回归均值框。
  按该教师四维归一化不确定性的均值，选择较低者作为回归目标；不增加 jitter 次数或教师前向。
- 有效候选须坐标/不确定性有限、宽高为正、不确定性非负，且与原融合框 IoU≥0.5。
  只有一个有效则选该教师；都无效或不确定性完全相同则回退原框。变换到学生视图后塌缩也回退。
  0.5 是固定几何对应保护，不是新伪标签接纳阈值，也不能证明候选属于同一真实目标。
- **先按原融合框完成 assignment/sampling，再替换已采样正 RoI 的回归目标。**
  原融合 NMS、平均不确定性筛选、RPN、分类目标、正负样本数量、bbox loss 分母都不变。
- `M3RoIHead` 不增加参数，只覆盖训练期回归目标构造。监督路径和推理路径继承原实现。
  M3 关闭默认回到原行为；`m3_target_mode=original` 用于开启路径的退化验收。
- 不叠加 M1、原始分恢复或框坐标投票。这里的均值是**同一候选的原有 jitter 均值**，
  不是对多候选做之前已经得到负结果的坐标投票。

主要文件：`ssod/models/dual_teacher.py`、`ssod/models/m3_routing.py`、
`ssod/models/roi_heads/m3_roi_head.py` 和独立 `phase3_dual_teacher_ssdd_dev_m3.py` 配置。
本次还把此前只在实验机存在的 M2/dev 配置纳入仓库；M2 配置增加独立输出目录以免覆盖旧 M0，
不改 M2 训练公式。所有模式实验必须显式使用新输出目录，不能用同一目录反复改模式训练。

## 2. 训练机：隔离拉取，不覆盖旧工程

下面在原训练机执行，保留原 conda `dt` 及 PyTorch/MMCV/MMDetection 版本。
旧工程可能包含未提交的 M2 或实验脚本，因此建议新克隆一个代码目录，复用数据和权重；
不要 `git reset --hard`，也不要执行仓库的安装脚本升级环境。

```bash
conda activate dt
(
  set -eu
  cd /home/xcc/dual_teacher_project
  test ! -e DualTeacher_m3
  git clone https://github.com/2gg-bit/cv.git DualTeacher_m3
  cd DualTeacher_m3
  for entry in data thirdparty ssdd_dev_protocol work_dirs; do
    test -d "/home/xcc/dual_teacher_project/DualTeacher/$entry"
    test ! -e "$entry"
    ln -s "/home/xcc/dual_teacher_project/DualTeacher/$entry" "$entry"
  done
)
cd /home/xcc/dual_teacher_project/DualTeacher_m3
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
git log -1 --oneline
git status --short
python -c 'import ssod, torch, mmcv, mmdet; print(ssod.__file__); print(torch.__version__, mmcv.__version__, mmdet.__version__); print(torch.cuda.is_available())'
nvidia-smi
df -h work_dirs
```

`ssod.__file__` 必须来自 **DualTeacher_m3**，不能指向旧工程。`thirdparty` 软链接用于
读取原版本 MMDetection 的继承配置，它未随本仓库上传。数据/协议/权重链接为共享存储：
旧文件只读使用，所有输出都写下面的新目录。链接可能出现在 git status 中，不提交它们。
若另一台机器没有原工程，先完整复制这些目录及原环境，再调整上述绝对路径；
GitHub 不含训练图像、开发划分或 checkpoint，单独 clone 不足以训练。

先确认预留足够磁盘。原 checkpoint hook 会另存 `latest.pth` 副本；不自动删除历史权重。
不要在已有训练占用 GPU 时并行跑下面的验收或训练。

## 3. 固定输入预检与严格初始化（三组）

下面 `selection_id` 仅访问既有路径，分别对应第一、第二、第三组**原有的图像选取**，
不是重新选图、重新划分或三个独立随机种子。保留同一开发集 186 张及每组 739 张无标签训练图。

```bash
python -m pytest -q tests
```

若原环境没有 pytest，先解决测试依赖；不要为此升级 PyTorch/MMCV。
本地 CPU 单测不替代下面真实权重与 CUDA 检查。

```bash
(
  set -eu
  cfg=configs/reproduce/phase3_dual_teacher_ssdd_dev_m3.py
  for selection_id in 6 7 8; do
    out="work_dirs/m3_v1/preflight/3/$selection_id"
    test ! -e "$out"
    python tools/check_m3_inputs.py "$cfg" --fold "$selection_id" --out-dir "$out"
    python tools/check_dual_teacher_init.py "$cfg" \
      --cfg-options fold="$selection_id" percent=3
  done
)
```

必须全部 PASS，原三张图与数据隔离不变，Phase1/2 对应组正确，严格加载四分支所有张量，
出现 `T1=S1, T2=S2, T1!=T2`。无需重训 Phase1/2。预检会记录文件哈希，但不替代实际加载。
开发集哈希已固定；失败时找配置或文件原因，不通过重新划分、跳过参数或放宽阈值解决。

## 4. 先做离线可行性审计，不训练

使用已经完成的三组 M2 最终权重检查“较低不确定性是否真的对应更好的框”。
这些权重**只用于诊断**，不作为后面 M3 正式训练的初始化。

```bash
(
  set -eu
  cfg=configs/reproduce/phase3_dual_teacher_ssdd_dev_m3.py
  for selection_id in 6 7 8; do
    ckpt="work_dirs/dev_ssdd/phase3_dual_teacher_m2_32000/3/$selection_id/iter_32000.pth"
    out="work_dirs/m3_v1/offline/3/$selection_id"
    test -f "$ckpt"
    test ! -e "$out"
    python tools/audit_m3_targets.py collect "$cfg" "$ckpt" \
      --fold "$selection_id" --seed 678 --out-dir "$out"
  done
)
```

脚本先封存不含 GT 输入的候选采集，再用开发集 GT 做独立分析；严格加载 full checkpoint。
每图保留原框、T1/T2 jitter 均值框、不确定性、选中来源、原回归接纳资格及真实原始教师检测分数。
不把原始教师分数伪装成 jitter 均值框的分数。输出 `candidates.json`、`metadata.json`、
`summary.json`、`summary.md` 等；可 CPU 复算，例如：

```bash
python tools/audit_m3_targets.py analyze work_dirs/m3_v1/offline/3/6 \
  --out-dir work_dirs/m3_v1/offline_reanalysis/3/6
```

重点检查：在**原本会参加回归的候选**上，相比原框和随机有效教师，
选择后同一 GT 的 IoU 是否改善、恶化多少、回退比例如何、各组/大小船是否一致。
T1/T2 固定选择和 GT oracle 仅是诊断对照，oracle 绝不进入训练；不以 GT 选择教师。
若无明显区分力、主要恶化或大量回退，先停在这里，不继续长训、不追加阈值搜索。
即使审计正向，也不能证明精度提升：测试预处理下的最终教师不等于训练早期的增强视图。

**到这里先保存并审查三组报告，不一口气执行后续长训。**

## 5. 真实训练批次的局部前向/反向验收

离线可行性通过后，使用第一组原有图像选取，验证同一真实 batch、同一状态与 RNG 的
M3关闭 / M3开启但用原框 / M3开启且选择教师三种路径。

```bash
python tools/check_m3_step.py configs/reproduce/phase3_dual_teacher_ssdd_dev_m3.py \
  --fold 6 --seed 678 --batch-index 0 \
  --out-dir work_dirs/m3_v1/step_init/3/6/batch0
```

默认严格从 Phase1/2 初始化，不执行 optimizer/EMA 更新。初始化早期可能没有正回归样本；
退出码 2 表示**未覆盖有效修改，不能当作通过**。预先限定按 batch-index 0→15 顺序检查，
每次用新目录；遇到失败(1)立即停、不能改容差凑通过。15 后仍未覆盖则保留“未覆盖”记录。
使用成熟 M2 checkpoint 可另外检查真实有效路由，三个组均做：

```bash
(
  set -eu
  for selection_id in 6 7 8; do
    python tools/check_m3_step.py configs/reproduce/phase3_dual_teacher_ssdd_dev_m3.py \
      --fold "$selection_id" --seed 678 --batch-index 0 \
      --checkpoint "work_dirs/dev_ssdd/phase3_dual_teacher_m2_32000/3/$selection_id/iter_32000.pth" \
      --out-dir "work_dirs/m3_v1/step_trained/3/$selection_id/batch0"
  done
)
```

同样遵守预声明的 batch 顺序，成熟权重验收不能替代初始化/短训检查。
必须实际改变正样本 bbox targets，原 GT、采样、标签/权重/负目标保持一致；
仅 `unsup1_loss_bbox` / `unsup2_loss_bbox` 允许改变，教师无梯度。
`report.json` 记录逐参数梯度比较与显式容差。该脚本是固定 loss-scale 的局部检查，
不模拟动态 AMP scaler/梯度裁剪/优化器/EMA；`--forward-only` 或 `--fp32` 只能补充诊断。

## 6. 独立目录做 100 iter 冒烟

```bash
(
  set -eu
  out=work_dirs/m3_v1/smoke/3/6
  test ! -e "$out"
  python -m torch.distributed.launch --nproc_per_node=1 \
    tools/train.py configs/reproduce/phase3_dual_teacher_ssdd_dev_m3.py \
    --launcher pytorch --seed 678 --no-validate --work-dir "$out" \
    --cfg-options fold=6 percent=3 runner.max_iters=100 \
    checkpoint_config.interval=100 log_config.interval=10
)
```

确认严格初始化 PASS、100/100、loss 有限、没有 OOM/NaN/持续非有限梯度，
记录 M2 保留比例、显存和耗时。任何瞬时 inf 均保留原始记录，不无依据归因于 warmup。
若动态 loss-scale/跳步未记录，则报告未知，不能写“AMP 已证实无溢出”。
不评估短训 AP、不用短训权重开始正式训练。保存日志和配置后先完成验收审查。

## 7. 验收通过才做三组正式 M3（同 seed 对照）

比较是 **M2 + 固定 Soft-NMS** 对 **M2 + M3 + 同一 Soft-NMS**。
不是与旧 seed=None 的 M0 比较，也不是把所有变化合成一个收益。
正式训练重新从对应 Phase1/2 开始，保持 seed=678、32000 iter；不从 M2 Phase3 或冒烟权重继续训。

```bash
(
  set -eu
  for selection_id in 6 7 8; do
    out="work_dirs/m3_v1/formal/3/$selection_id"
    test ! -e "$out"
  done
  for selection_id in 6 7 8; do
    out="work_dirs/m3_v1/formal/3/$selection_id"
    python -m torch.distributed.launch --nproc_per_node=1 \
      tools/train.py configs/reproduce/phase3_dual_teacher_ssdd_dev_m3.py \
      --launcher pytorch --seed 678 --no-validate --work-dir "$out" \
      --cfg-options fold="$selection_id" percent=3
    test -f "$out/iter_32000.pth"
    sha256sum "$out/iter_32000.pth"
  done
)
```

顺序执行，任何异常即停；不边看 AP 边改配置或选最佳 checkpoint。
本命令不启动开发集评估，不访问官方测试集。源码本身未引入自动监控任务。

## 8. 完成后评估与交付

仍用原来的 186 张开发集固定 A/B 评估脚本，输入换为 M3 解析配置和最终 `iter_32000.pth`，
输出写 `work_dirs/m3_v1/eval/...` 新目录。必须使用本仓库代码并严格加载完整参数，
只推理 teacher2；Hard-NMS 与固定 Linear Soft-NMS 都报告。
Soft-NMS 仍是已冻结的评估流程，不是本 M3 配置自动启用的推理算子。

本仓库旧 `tools/eval_teacher2_export.py` 针对官方 232 张测试集写有专用校验，
**不能直接拿它评估这轮 186 张开发集**；不要为了跑通临时改测试集或放宽覆盖检查。
开发集 A/B 的本地脚本与候选缓存工具位于原实验材料中，未作为本次 M3 修改重写。

交付：离线三组审计、GPU `report.json`、100 iter 日志、三组正式完整日志/配置/启动命令、
代码 revision 与 diff 状态、Phase1/2/最终权重 SHA256，以及逐框预测、完整 COCO 指标与评估元数据。
主看三组配对 AP；同时报告 AP50/AP75/AP85、AR、小/中/大船以及背景/重复，不只列正结果。
额外种子仍须 M2/M3 配对，不把三组原有图像选取当作三次独立种子复现。

已有 M2 正向结果只支持继续研究，不保证 M3 有效。若 M3 负向，保留负结果、回到冻结 M2；
不要立即叠加其它部件。开发集已经参与方法开发，不能再称其未接触测试集；
论文也须披露使用了额外开发标注，不能声称整个开发过程仅用了三张 SAR 标注图。
