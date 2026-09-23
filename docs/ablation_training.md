# 正确复现后的四组独立实验

此前所有旧实验结果（包括旧复现、M2、FG、MVDT 等）作废，不用于本轮选型、调参或判断增益。本轮参照用户已经完成的正确复现 B0。B0 无须重新训练；如果只有训练结果而没有可比的预测导出，可仅重新评估其正确 checkpoint。

## 代码范围

保留 Phase1 → T1/S1、Phase2 → T2/S2 的严格初始化，四分支 checkpoint 恢复、teacher EMA、原始普通 NMS 伪标签融合（fusion IoU=0）。学习率、数据划分、训练轮数和推理设置由 B0 继承。

删除 M1 定位质量头、M3 回归目标路由、MVDT 动态阈值及其旧组合配置、启动器和专属测试。旧版本仍可从 Git 历史恢复。没有删除仓库外的资料、训练权重或日志。“移除”代表暂不纳入本轮实验，不能用失效结果断言这些方法无效。

保留 M2 和 FG，新增 PG，并拆成以下独立对照。默认全关闭，原 B0 配置仍然可用。

| 顺序 | 配置名 | 唯一实验改动 |
|---|---|---|
| E1 | m2.py | 无监督正 RoI 分类权重乘 stopgrad(min(p_T1,p_T2))；背景权重、原归一化分母 Z0、回归损失不改 |
| E2 | fg.py | 仅真标注 sup1/sup2 的 P2 前景辅助监督，内部权重 0.1，M2 关闭 |
| E3 | pg_sup2.py | 仅有标注 SAR 损失系数渐增 |
| E4 | pg_both.py | 有标注和无标注 SAR 损失系数同时渐增 |

PG 是本项目的待验证消融方案，不等于完整复现 D3T 的 zigzag 策略。固定计划为：
`r(t) = 0.5 + 0.5*t/(T-1)`，`t=0,...,T-1`，T 继承 B0 的 max_iters。
原系数为 gamma=0.2、alpha=2 时：

- B0/E1/E2：sup2=0.2，unsup2=0.4。
- E3：sup2 从 0.1 线性升到 0.2；unsup2 恒为 0.4。
- E4：sup2 从 0.1 升到 0.2；unsup2 从 0.2 升到 0.4。
- sup1=1、unsup1=alpha 均保持原值。若正确 B0 明确配置了不同的 sup2_weight/unsup_weight，以 B0 为准。

PG 由 runner.iter 驱动，额外 forward 不推进进度；checkpoint 保存其设置和位置。FP16 不降低调度状态精度。继续训练必须使用匹配配置和 --resume-from，不能把 PG checkpoint 当作普通 B0 初始化。

## 1. 安装到训练机

优先将离线包中的 repository.bundle 克隆到新目录，避免把已删除的旧模块留在覆盖目录中。沿用正确复现的 conda 环境，不在此阶段升级 torch/mmcv/mmdet。

```bash
# 在解压后的离线包目录执行；无需联网
sha256sum -c SHA256SUMS.txt
git clone repository.bundle /你的路径/DualTeacher_ablation
cd /你的路径/DualTeacher_ablation
git rev-parse HEAD
conda activate dt
```

克隆目录不包含数据集、Phase1/2、第三方环境或 CUDA。把旧正确仓库的 data、work_dirs 挂接进新目录，使 B0 的相对路径保持原义：

```bash
# OLD 必须是已正确复现的仓库，不是旧错误实验目录
OLD=/你的路径/正确复现仓库
ln -s "$OLD/data" data
ln -s "$OLD/work_dirs" work_dirs
# configs/dual_teacher/base.py 的继承路径要求 thirdparty 位于仓库的上一级。
# 将新 checkout 放在正确旧仓库旁边，可以共享 ../thirdparty/mmdetection。
test -f ../thirdparty/mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py
```

这些链接只供共享已有输入和写入新的实验子目录；不覆盖原运行。若 thirdparty 检查失败，先把正确旧仓库旁的 thirdparty 接到新仓库的上一级，不重新下载其他版本。mmdet 和 backbone 的预训练缓存沿用正确复现环境；它们不在代码包内。不要对旧目录执行解压覆盖，也不要重新运行旧的跨 seed 编排器。

```bash
python tools/train_ablation.py --check-source-only
```

检查每条 [Ablation source] 的 repo_root 和 ssod 源码路径都属于新 checkout。入口会给真正执行训练的 worker 固定 sys.path/PYTHONPATH 并断言导入路径、记录哈希；不是只检查 launcher。

## 2. 冻结实际 B0 并生成配置

所有变量必须来自这次正确复现记录，不沿用旧报告中的 seed/fold。下面采用仓库标准 B0 路径；若训练时做过额外 cfg-options，请逐项照抄，或直接传入正确运行保存的 resolved config。

```bash
# 先按正确复现记录设置这三个值，例如 export B0_FOLD=...
: "${B0_FOLD:?填写正确 B0 的 fold}"
: "${B0_PERCENT:?填写正确 B0 的 percent}"
: "${B0_SEED:?填写正确 B0 的 seed}"
B0=configs/reproduce/phase3_dual_teacher_ssdd.py
SUITE="ablation_configs/fold${B0_FOLD}_seed${B0_SEED}"
WORK="work_dirs/ablation_v1/fold${B0_FOLD}_seed${B0_SEED}"
python tools/prepare_ablation_suite.py "$B0" \
  --out-dir "$SUITE" --work-root "$WORK" --seed "$B0_SEED" \
  --cfg-options fold="$B0_FOLD" percent="$B0_PERCENT"
```

输出 b0.py（对照快照）及四组实验配置、baseline_resolved.py、manifest.json。manifest 记录 B0、所有配置、Phase1/2 和数据标注文件的 SHA256，以及逐项配置差异。生成器不会训练，也不会改 B0。

核对差异只涉及模块开关、新 work_dir、显式 seed 和禁用自动恢复。数据、增强、采样比例、学习率、EMA、迭代次数、fp16、评价口径应与正确 B0 一致。out-dir/work-root 必须全新，不能复用已经存在的运行目录。四个 configs/reproduce 模板可供查阅；正式训练优先使用生成的完整配置。

## 3. 每个实验先验收

在训练机 GPU 上执行；从 E1 开始，将 EXP 依次改为 m2、fg、pg_sup2、pg_both。

```bash
set -euo pipefail
EXP=m2
python tools/train_ablation.py --check-init "$SUITE/$EXP.py" \
  2>&1 | tee "$SUITE/${EXP}_init.log"
python tools/train_ablation.py --check-step "$SUITE/$EXP.py" \
  --seed "$B0_SEED" --out-dir "$SUITE/${EXP}_acceptance" \
  2>&1 | tee "$SUITE/${EXP}_acceptance.log"
```

CPU 初始化必须出现 T1=S1、T2=S2、T1!=T2；CUDA 验收 result.json 必须为 passed。验收使用真实训练批次，不读验证 AP，不做 optimizer/EMA 更新。

- M2：off/on/强制权重为 1 回放；两个无监督分支均需覆盖降权正例，并验证其他损失及恒等设置。
- FG：原检测损失不变，只新增 sup1/sup2 前景损失；关闭时辅助头无梯度，开启时梯度非零且有限。
- PG：起点只按计划缩放目标分支，终点还原 B0；teacher 不求梯度。
- 如果 M2 首批没有覆盖正例，结果为 incomplete、退出码 2；使用新的 out-dir 和 --batch-index 1、2…重试。不能将“未覆盖”记成通过。
- 出现非有限值或不符合损失路由就停止，先定位代码/环境问题，不以改阈值来绕过验收。

## 4. 完整训练

每组从同一正确 Phase1/Phase2 新建 Phase3，不能接 B0、M2 或 FG 的 Phase3 checkpoint 继续训练。固定 seed、fold、T 和 GPU 数量；本示例为单卡，需与正确 B0 一致。若正确 B0 使用 --deterministic 或 --no-validate，训练命令也保持一致。

```bash
set +e
python -m torch.distributed.launch --nproc_per_node=1 \
  tools/train_ablation.py "$SUITE/$EXP.py" \
  --launcher pytorch --seed "$B0_SEED" \
  2>&1 | tee "$SUITE/${EXP}_train.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$TRAIN_STATUS" > "$SUITE/${EXP}_train_exit_code.txt"
test "$TRAIN_STATUS" -eq 0
```

记录最终 iter_T.pth 的 sha256，核对最后一步 T/T。PG 日志 [PG weights] 应从对应起点到终点。退出成功和 CUDA 验收不能替代最终训练日志检查。动态 loss scaler 的净缩放次数、采样到的 inf 条数都不能直接当作真实跳步率。

训练被中断时，只能显式添加 --resume-from "$WORK/$EXP/iter_N.pth" 恢复该组；PG 会核对配置和 runner 进度。不要设 load_from 或 auto_resume。

本机未运行真实 CUDA / MMDetection 验收，也未开始上述训练。交付包附本机 CPU 测试记录。

## 5. 同口径评估与下一轮

T 以生成配置中的 runner.max_iters 为准，不能为了追求 AP 临时延长某一组训练。若正确 B0 使用当前仓库标准的 232 图 test.json / teacher2 评价，可以使用：

```bash
# T 填生成配置中的 max_iters
: "${T:?填写该组 max_iters}"
set -o pipefail
python tools/train_ablation.py --eval "$SUITE/$EXP.py" \
  "$WORK/$EXP/iter_${T}.pth" --fold "$B0_FOLD" \
  --out-dir "$SUITE/${EXP}_evaluation" \
  2>&1 | tee "$SUITE/${EXP}_evaluation.log"
```

导出器保留配置中的 score_thr、NMS、max_per_img，固定使用 teacher2 和完整 232 图 test.json，不会自动切成旧的 186 图 dev。若正确 B0 的评价脚本或数据不同，须使用其原评价流程，不强行套用本导出器。PG/FG 不改变推理打分规则，FG 辅助头不参与检测输出。

先完成同一正确 B0 设置下的 E1→E2→E3→E4，每组填以下表格。主指标采用 B0 的原 mAP 定义，另外记录 AP50/AP75/APs/AR 和原评价流程支持的误检分档，不把 AP50 写成 COCO mAP。

| 实验 | B0 mAP | 本组 mAP | 配对 ΔmAP | AP50 | AP75 | APs | AR |
|---|---|---|---|---|---|---|---|
| E1 M2 | | | | | | | |
| E2 FG | | | | | | | |
| E3 PG-sup2 | | | | | | | |
| E4 PG-both | | | | | | | |

下一阶段对值得继续验证的单模块扩展其他预定 fold/seed，与每个设置下的正确 B0 配对；缺少正确 B0 的设置先补 B0。对各折等权平均配对 Δ，不因绝对 AP 低而删折。组合模块实验放在单模块对照之后。论文最终测试集不能反复用于挑系数；本轮固定 FG=0.1、PG 起点比例=0.5，不根据测试 AP 临时改动。上述数值只是首轮实验设置，尚无新有效结果证明能提高精度。
