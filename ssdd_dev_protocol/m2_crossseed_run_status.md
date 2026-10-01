# M2 跨种子复验 — 12 次运行状态表

- 协议：见 `DualTeacher_m3/M2_crossseed_protocol.md`（冻结）
- 编排脚本（**当前生效：v3**）：`ssdd_dev_protocol/crossseed_orchestrator.py`
  （SHA256 `bf379e021bcda1278f0dc3474f9fafda7b36ba33d5805b992a550f12b0d02ecc`），
  运行于 screen 会话 `m2crossseed`（PID 2442194），2026-09-19 18:19:46 起接管。
  - 顺序：**训练 → 自动核验 → SHA256 入表 → 立即启动下一次**；串行。
  - 核验失败或训练非 0 退出 → 停止队列并告警；**不以 AP 评估作为放行条件**。
  - 放行分两类（见下节"A/B 两类放行"）。
- 前一版编排 `run_m2_crossseed_v2.sh`（SHA256 `75a6ca9bf37dd9f99b8dd435bc93ebac212d4977d0aaed1f38062cc039edd67d`，现已退役）：
  **2026-09-19 18:15 仅终止该 bash 编排进程（PID 2381946）**，
  未触及训练进程（PID 2319392/2319534 属独立会话 sid/pgid `2319390`，父进程已是 systemd）。
  终止后确认训练仍在推进（iter 5050→5150），再启动 v3。
- 首版脚本 `run_m2_crossseed.sh`（SHA256 `2e747a7b…`）已退役：它无核验 gate。2026-09-19 17:31 在训练交接处切换
  ——仅终止 launcher 进程（PID 2319390），**当前训练进程未中断**（迭代 2100→2200→2700 持续推进），确认后启动 v2。
- 核验脚本：`tools/verify_m2_crossseed_ckpt.py`（SHA256 `4afd815e593c92583e0c196ea072fedfbc8eea6556b1f9d46b30dd8fb788649f`）
- 运行清单（机器可读）：`ssdd_dev_protocol/m2_crossseed_runs.tsv`
  - v2 的表头为 9 列；**gate v3 的表头为 11 列**（增加 `config_snapshot_sha256`、`source_sha256`）。
    `ledger_append()` 每次以 `LEDGER_COLS` 重写整个文件，故 v3 首次写入时表头自动升为 11 列；
    当前表中只有表头、无数据行，升级不丢记录。

## 第二轮修订（gate v3，2026-09-19）

**注意：当前正在执行的仍是 v2（PID 2381946）**，v3 尚未接管编排。v3 一律另存为新文件名，
**未覆盖正在运行的 v2 shell 文件**。切换将在 run #1 训练结束的交接点进行。

| 文件 | SHA256 | 作用 |
|------|--------|------|
| `tools/crossseed_gate.py` | `26d49e4197c872e7a65c40194fa0833d309351d61513262926935bf982151cd7` | 可测纯函数库：身份核验、幂等跳过、批准绑定、账本、原子写 |
| `tools/crossseed_verify.py` | `27bd643b5ff6e7a5f69aeff83235db9206f81cbbd75b9099138c0a0595f35bd5` | 单次运行核验 CLI（CPU，无数据集/GPU/AP） |
| `ssdd_dev_protocol/crossseed_orchestrator.py` | `bf379e021bcda1278f0dc3474f9fafda7b36ba33d5805b992a550f12b0d02ecc` | Python 编排器 v3（替代 v2 bash） |
| `tools/test_crossseed_gate.py` | `547e112e6e1cc16ff69af97bca8eddcc8245cc5bf393a17734b31a20a0e73379` | 44 项 CPU 故障测试 |

### v3 相对 v2 修好的问题

1. **权重必须属于"这次实验"**：不再用"当前配置能加载权重"作证明（M0/M2 参数结构相同，该证明无效）。
   改为从 **checkpoint 内嵌的 resolved 配置**抽取身份字段，与冻结期望卡逐项比对：
   `fold`（原有图像选取）、`percent`、`runner_max_iters`、`max_keep_ckpts`、`model.type`、
   `model.model.type`、`load1_from`/`load2_from`（初始化路径）、`auto_resume`、
   `load_from`/`resume_from`、`m2_enabled`、`m2_force_weight_one`、`m3_enabled`；
   另加 `meta.seed`（**缺失即判失败**）、`meta.exp_name`（M0/M2 身份）与 `meta.iter` 完成度。
2. **与启动快照交叉核对**：再解析 run 目录里 mmcv 在启动时写下的配置快照，逐字段比对；
   checkpoint 内嵌配置与启动快照不一致 → 拒绝。
3. **证据分级，不冒充**：配置快照标 `written_by_mmcv_at_run_start`；
   源码与 Phase1/2 初始化权重的哈希是**核验当时**的文件，标 `verification_time_current_file`，
   不称为"启动时证据"。
4. **记录失败即停队列**：核验结果、`VERIFIED.json`、账本全部用
   **临时文件 + `fsync` + `os.replace`** 原子落盘；任一失败 → `die`，不再启动下一次。
5. **重启可恢复**：启动时先 `reconcile()` —— 若 `VERIFIED.json` 已写而账本缺行，
   按其中的产物哈希补齐，幂等（重复重启不产生重复行、不重复训练）。
6. **人工批准绑定具体产物**：请求 ID = `sha256(run_key, seed, ver, fold, checkpoint_sha256,
   config_snapshot_sha256, source_sha256)`；放行文件必须携带与当前请求一致的全部哈希，
   **旧批准不能放行新权重**，也不再只判断占位文件是否存在。
7. **交接例外仅限 run #1**：`HANDOVER_RUNS = {seed123/m0/3/6}`；
   其余运行退出码丢失（`None`）→ 停队列排查，**不会**自动转成"交接放行"。
8. **训练进程检查绑定到具体运行**：按 `/proc/<pid>/cmdline` 匹配 `tools/train.py` **且**
   `--work-dir` 等于该 run 目录，才认为它在跑；每次启动前检查输出盘余量（<20GB 停）。
9. **中断后重启不会重复训练**：目录中只有我们自己的 `run_manifest.json`（或空）→ 说明上次在训练
   启动前就失败了，允许重新发起；一旦出现 mmcv 日志或任何 `.pth` → 停队列排查，不自动重跑。

## A/B 两类放行

**(A) 由 v2 启动的 run（#2–#12）**：真实退出码必须为 0，且全部核验通过 → **自动放行**，无需人工确认。

**(B) 交接 run（仅 #1）**：它的父 launcher 已被终止，**真实退出码不可得**。处理方式：

- `exit_code=null`、`exit_status=unknown_due_to_handover`，**不伪造退出码 0**。
- 训练结束后仍执行完整核验（日志完成度 + 最终 checkpoint + 严格加载 + 哈希）。
- 核验通过后写 `HANDOVER_RELEASE_REQUIRED.json` 并**等待一次性人工放行**：由人创建
  `<run_dir>/HANDOVER_RELEASED` 后队列才继续；放行结果写入 `VERIFIED.json` 后删除该占位文件。
- 说明：**旧 launcher 的 `DONE` 行不能作为代理判据**——launcher 已被终止，它不会在训练结束时再打印该行，
  `set -e` 也不改变这一点。该判据已移除。

## 每次完成的核验 gate（`verify_m2_crossseed_ckpt.py`，CPU、无数据集、无 GPU、无 AP）

1. 退出码 0 —— 仅 (A) 类可判定；(B) 类记为 `unknown_due_to_handover`
2. 日志完成 `32000/32000`（读 `*.log.json` 最后一条 train 记录）
3. checkpoint 可读，且 meta 与训练记录一致
   - 注意：`meta['iter']` 是 0-indexed 的 `runner.iter`，故 `iter_32000.pth` 对应 `meta.iter=31999`。
     已在 Phase1（7999/8000）、Phase2（11199/11200）、Phase3（31999/32000）上核对该固定关系。
   - `meta['seed']` 必须等于运行种子
4. checkpoint 内**全部浮点张量有限**（2616 个 state 张量）
5. **按对应配置严格加载**：`build_detector(cfg.model)` 后 `load_state_dict(strict=True)`，
   再检查模型全部参数/缓冲区有限（2200 个浮点张量）。**仅 `torch.load` 成功不视为通过。**
6. 通过后写 `VERIFIED.json` + 追加 `m2_crossseed_runs.tsv`，并检查下一次运行的磁盘余量（<10GB 停队列）

### 幂等跳过条件（不只是"标记存在"）

`verified_ok()` 要求 `VERIFIED.json` 的 `run_dir / seed / ver / fold / iters` 与当前目标一致，
**且其中 `sha256` 与磁盘上 `iter_32000.pth` 的实际哈希相同**，否则视为未核验（防止标记仍在而权重缺失或被替换）。
已用 5 个用例验证：无标记→1、匹配→0、fold 不符→1、seed 不符→1、sha256 过期→1。

**dry-run 已验证**：对 seed=678 M2/fold6 权重核验 PASS，复现记录的 SHA256 `37d29bed…`；种子不符时退出码 1。
- 运行目录根：`DualTeacher_m3/work_dirs/m2_crossseed/`
- 公共参数：`--seed {123|456} --no-validate`、`fold={6|7|8} percent=3`、`auto_resume=False`、
  `runner.max_iters=32000`、`checkpoint_config.max_keep_ckpts=1`（只留 iter_32000.pth + latest.pth 副本）
- **保存策略调整（适用全部 12 次）**：`checkpoint_config.max_keep_ckpts=1`。
  - 效果：iter 32000 落盘后删除 iter_4000…28000；最终保留 `iter_32000.pth` + `latest.pth`（`shutil.copy` 真实副本）。
  - **注意**：这是**预计保留量**，不是严格磁盘峰值。CheckpointHook 先保存新权重、再删除旧快照（[源码 v1.3.9](https://raw.githubusercontent.com/open-mmlab/mmcv/v1.3.9/mmcv/runner/hooks/checkpoint.py)），保存期间可能同时存在新旧权重与 latest；另有日志与评估产物。
  - 12 次最终保留量估算 ≈ **34.8GB**（12 × 2.9GB），另需预留保存瞬时空间。
- 每 run 计时约 8.9 h（32000 × ~1.0 s/iter）；12 run 串行约 4.5 天
- 判定：`ΔAP = AP_M2 − AP_M0`，逐对报告；seed=678 单列，不与本次混算

## 状态（按执行顺序）

| # | seed | ver | fold | 目录 | 状态 | start | done | iter_32000 SHA256 |
|---|------|-----|------|------|------|-------|------|-------------------|
| 1 | 123 | m0 | 6 | `seed123/m0/3/6` | RUNNING | 2026-09-19 16:56:59 | — | — |
| 2 | 123 | m2 | 6 | `seed123/m2/3/6` | PENDING | — | — | — |
| 3 | 123 | m0 | 7 | `seed123/m0/3/7` | PENDING | — | — | — |
| 4 | 123 | m2 | 7 | `seed123/m2/3/7` | PENDING | — | — | — |
| 5 | 123 | m0 | 8 | `seed123/m0/3/8` | PENDING | — | — | — |
| 6 | 123 | m2 | 8 | `seed123/m2/3/8` | PENDING | — | — | — |
| 7 | 456 | m0 | 6 | `seed456/m0/3/6` | PENDING | — | — | — |
| 8 | 456 | m2 | 6 | `seed456/m2/3/6` | PENDING | — | — | — |
| 9 | 456 | m0 | 7 | `seed456/m0/3/7` | PENDING | — | — | — |
| 10 | 456 | m2 | 7 | `seed456/m2/3/7` | PENDING | — | — | — |
| 11 | 456 | m0 | 8 | `seed456/m0/3/8` | PENDING | — | — | — |
| 12 | 456 | m2 | 8 | `seed456/m2/3/8` | PENDING | — | — | — |

## 启动前核验（run #1，2026-09-19）

- 源码：`DualTeacher_m3` @ `572f788`，`ssod/models/dual_teacher.py` SHA256 `8e79bcf6…`（= 提交版本，与 `DualTeacher` 一致）
- 种子：`Set random seed to 123, deterministic: False` ✅
- 四分支严格初始化 ✅
  - `phase1_pretrain_optical/100/1/iter_8000.pth` → teacher1 / student1（各 verified 654 state tensors, strict）
  - `dev_ssdd/phase2_pretrain_optical_sar/3/6/iter_11200.pth` → teacher2 / student2（各 verified 654 state tensors, strict）
  - `PASS: T1=S1, T2=S2, T1!=T2; fusion=NMS, fusion_iou=0`
- 开关：M0 resolved 配置无 `m2_enabled`（默认 False）、无 `m3_enabled`（M3 关闭）✅
- `auto_resume=False`、`runner.max_iters=32000`、`checkpoint_config.max_keep_ckpts=1` ✅
- 磁盘启动时 60GB 可用；峰值占用 ≈ 2.9GB/run

## 健康巡检记录

| 时间 | run | iter | loss | grad_norm | 显存(MB) | s/iter | 备注 |
|------|-----|------|------|-----------|----------|--------|------|
## 关于非有限梯度范数（grad_norm=Infinity）的观测记录

**事实（不做因果归因）：**

- 本轮 run #1 在 Iter[50]、Iter[400] 各出现一次 `grad_norm=Infinity`，相邻 Iter 已恢复有限；全程 loss 有限。
- **冻结基线（seed=678）全部 6 次运行（M0×3、M2×3）均出现同一现象**，各 **12–15 条日志记录含非有限梯度范数**
  （该计数是**日志记录条数**，不等于实际溢出次数或优化器跳步次数，二者无法从现有日志判定）。例如 M0/fold6 出现在 iter
  `50, 400, 2700, 4900, 8300, 10350, 14400, 16550, 18600, 22000, 26000, 27300, 31300`（640 条训练记录中 13 条）。
- 该现象**不局限于 warmup**（分散于全程、非周期），6 次运行均跑满 32000 并产出可用权重；**出现时 loss 仍为有限值**。
- 上述仅说明它**不是本轮特有、也不限于 warmup**，**不能据此认定无害**。
- **原因未确认**：没有逐步 grad-scale / 跳步记录，无法判定成因或跳步次数。不中途修改 AMP 或学习率。

**判定与巡检口径**：这是冻结配置的既有行为，本轮未见新增异常，继续观察。
巡检条件**包含持续非有限 `grad_norm`**：即使 loss 仍有限，只要出现连续多次非有限 `grad_norm` 就提醒。
M0 与 M2 的异常情况**分别记录**（不合并统计）。

## 健康巡检记录

| 时间 | run | iter | loss | grad_norm | 显存(MB) | s/iter | 备注 |
|------|-----|------|------|-----------|----------|--------|------|
| 2026-09-19 16:59 | 1 | 100 | 0.550 | 7.44 | 9244 | 0.93 | Iter[50] `grad_norm=Infinity`，Iter[100] 已恢复有限，原因未确认 |
| 2026-09-19 17:35 | 1 | 1700 | 0.900 | 7.15 | 9887 | 0.92 | Iter[50]/[400] 各一次 `grad_norm=Infinity`；loss 全程有限；见上节 |
| 2026-09-19 18:16 | 1 | 4950 | 0.998 | 6.99 | 11545 | 0.94 | 非有限 `grad_norm` 出现在 Iter 50/400/2900（不连续，非持续）；GPU 利用率 99% |
| 2026-09-19 18:21 | 1 | 5300 | 0.855 | 6.49 | 12399 | 0.94 | 非有限 `grad_norm` 增至 Iter 50/400/2900/5200 —— 均为孤立单次，**无连续多次**，loss 全程有限；编排已由 v3 接管 |

## 故障测试结果（CPU，合成用例）

`python tools/test_crossseed_gate.py` → **cases: 44   pass: 44   fail: 0**（退出码 0）
完整输出：`ssdd_dev_protocol/real_verify_seed678/crossseed_gate_fault_tests.log`
**未占用训练 GPU**（纯逻辑 + 临时目录小文件，无 checkpoint、无数据集）。

| 要求 | 用例 | 结果 |
|------|------|------|
| seed/配置/权重正确 → 通过 | `1`, `1b`(×2) | PASS |
| seed 缺失或与期望不符 → 拒绝 | `2a` 缺失 / `2b` None / `2c` 不符 / `2d` 日志种子矛盾 | PASS ×4 |
| 原有图像选取或 M0/M2 身份不符 → 拒绝 | `3a` fold / `3b` M0 冒充 M2 / `3c` M2 冒充 M0 / `3d` exp_name / `3e` M3 开启 / `3f` 初始化路径 / `3g` 提前结束 | PASS ×7 |
| 权重或配置被改而旧 VERIFIED 仍在 → 拒绝跳过 | `4a` 未变仍跳过 / `4b` 权重被替换 / `4c` 快照被编辑 | PASS ×3 |
| 人工批准属于旧请求或旧权重 → 拒绝放行 | `5a` 匹配放行 / `5b` 旧 request_id / `5c` 不同权重哈希 / `5d` 不同 run / `5e` 缺哈希 / `5f` 空占位文件不是批准 | PASS ×6 |
| 写入失败 / 空间不足 → 停队列，不启动下一次 | `6a` 只读目录抛错 / `6b` 不留 `.tmp` 残片 / `6c` 空间不足 / `6d` 空间充足 | PASS ×4 |
| 写入中崩溃后重启 → 对账，不重复训练、不漏记录 | `7a`–`7e` | PASS ×5 |
| 之后某次非 0 退出或退出码未知 → 停队列，不降级放行 | `8a` 0 / `8b` 1 / `8c` -9 / `8d` None(自建) / `8e` 交接 / `8f` | PASS ×6 |
| （补充）崩溃在训练启动前 → 可重启，不算重复训练 | `9a`/`9b` 仅 manifest → 可重启；`9c` 有 mmcv 日志 / `9d` 有部分权重 → 停队列 | PASS ×4 |
| （补充）内嵌配置与启动快照不一致 → 拒绝 | `3h` 不一致拒绝 / `3i` 一致通过 | PASS ×2 |

## 真实权重核验报告（CPU，seed=678 已完成权重）

**与上面的合成测试分开报告。** 用了 `DualTeacher/work_dirs/dev_ssdd/` 下 seed=678 的两份最终权重：

| 运行 | 目录 | `iter_32000.pth` SHA256 | 大小 |
|------|------|--------------------------|------|
| seed678 M0/fold6 | `dev_ssdd/phase3_dual_teacher/3/6` | `e03857138f773470a609b2c9052dfcf44ec7ba56036173944ddca3389d698077` | 1.448 GB |
| seed678 M2/fold6 | `dev_ssdd/phase3_dual_teacher_m2_32000/3/6` | `37d29bedc32adad05fe2e29c7869b9f9d4ed340ed836f3ea7b22724ff8dbd1a1` | 1.448 GB |

M2 的哈希与上一轮记录的 `37d29bed…` **逐位重现**。两份都 **VERIFY PASS（退出码 0）**：

- 日志最后一条 train 记录 `32000`；日志种子 `678`；`meta.iter=31999`（0-indexed runner.iter）、
  `meta.epoch=5`、`meta.seed=678`；`meta.exp_name` 分别为
  `phase3_dual_teacher_ssdd_dev.py` / `phase3_dual_teacher_ssdd_dev_m2.py`
- 2616 个 state 张量全部有限；按 checkpoint 内嵌配置 `build_detector` 后 `load_state_dict(strict=True)`
  成功，2200 个浮点参数/缓冲区全部有限
- 内嵌配置与启动快照（`written_by_mmcv_at_run_start`）逐字段一致
- 两个 Phase1/2 初始化路径的哈希两 run 一致：phase1 `70627d7eee2a55f5…`、phase2 `7b73dccfa5a05414…`
  （标签 `verification_time_current_file`）
- 报告：`ssdd_dev_protocol/real_verify_seed678/{m0,m2}_fold6_verify_result.json`

**负面用例（同一批真实权重，错误的身份声明）——全部拒绝：**

| 用例 | 结果 | 拒绝理由 |
|------|------|----------|
| N1 M0 权重声明为 M2 | 拒绝 | `exp_name` 与 `m2_enabled=False` 均不符 |
| N2 M2 权重声明为 M0 | 拒绝 | `exp_name` 与 `m2_enabled=True` 均不符 |
| N3 M2 权重 + 错误种子 123 | 拒绝 | `meta.seed=678 != 123`，日志种子亦为 678 |
| N4 M2/fold6 权重声明为 fold 7 | 拒绝 | `fold=6 != 7`，且 `load2_from` 指向 fold6 的 Phase2 权重 |

另：这两份 seed678 权重在**默认期望（`max_keep_ckpts=1`，本轮策略）**下被正确拒绝——
`config max_keep_ckpts=10 != expected 1 (checkpoint retention policy)`，
说明"保存策略"也构成身份的一部分（seed678 冻结批用的是 10）。核验时按该批实际策略传 `--expected-max-keep-ckpts 10`。

## 编排器 dry-run

`python ssdd_dev_protocol/crossseed_orchestrator.py --dry-run` → 退出码 0：
`reconcile: ledger already consistent with VERIFIED.json files`；
`seed123/m0/3/6` 判为 `handover-pending`，其余 11 项 `pending`；**未启动任何训练**。

## 编排接管记录（v2 → v3）

**切换前确认的实际进程关系：**

| 角色 | PID | pgid / sid | 父进程 | 说明 |
|------|-----|-----------|--------|------|
| 训练 launcher | 2319392 | 2319390 / 2319390 | systemd (2182) | 原 launcher 已不在，父进程已交 systemd |
| 训练 worker | 2319534 | 2319390 / 2319390 | 2319392 | `STAT=Rl`，持续推进 |
| v2 编排 | 2381946 | 2381946 / 2381946 | — | 独立会话，子进程仅 `sleep 60` |

训练进程与 v2 **不在同一会话、不在同一进程组**，因此终止 v2 不可能波及训练。

**切换步骤：**

1. 2026-09-19 18:15 对 v2 仅发 `SIGTERM`（PID 2381946）；确认其已退出。
2. 确认训练进程仍在运行（`STAT=Rl`，iter 推进 5050→5150），**未重启、未改参数**。
3. 2026-09-19 18:19:46 在 screen 会话 `m2crossseed` 启动 v3。
4. v3 立即按 `--work-dir` 精确匹配绑定到在飞的 run #1，日志：
   `handover: seed123/m0/3/6 is in flight (pids [2319392, 2319534]); waiting`。

**关于 18:16:17 的一次瞬时运行（PID 2436811）——如实记录：**

一次 v3 启动尝试（18:16:17）在被取消之前已经实际启动，打印到
`handover: ... is in flight; waiting` 后被终止。经核查它**只写了日志行**，
未产生任何产物：无 `VERIFIED.json`、无 `verify_result.json`、无 `run_manifest.json`、
无 `HANDOVER_RELEASE_REQUIRED.json`，账本仍只有表头（1 行）。无残留子进程。
该次中断不影响实验记录，也不影响当前运行。

**当前状态：** 仅一个编排进程（PID 2442194），正在等待 run #1 训练结束。

**查看方式：** `screen -r m2crossseed`（脱离用 `Ctrl-A` 然后 `D`）；
日志文件 `ssdd_dev_protocol/crossseed_orchestrator.log`。

**run #1 结束后的动作：** v3 会自动执行完整核验；通过后写
`work_dirs/m2_crossseed/seed123/m0/3/6/HANDOVER_RELEASE_REQUIRED.json` 并等待人工放行。
**人工放行需由人创建** `HANDOVER_RELEASED`（内容须携带与请求一致的
`request_id`、`checkpoint_sha256`、`config_snapshot_sha256`、`source_sha256`、`run_key`）。
放行前 **不会** 创建该文件，也 **不会** 自动开始 run #2。

---

## v4 新版本修复与隔离测试（2026-09-19，未切换编排）

**本轮范围：** 只写新文件、只做 CPU 隔离测试与真实权重核验。
**未** 切换编排、**未** 创建 `HANDOVER_RELEASED`、**未** 改动 v3 文件、**未** 操作训练。

**新文件（与 v3 并存，v3 仍在驱动训练）：**

| 文件 | SHA256 |
|------|--------|
| `tools/crossseed_gate_v4.py` | `b37c4a725b583c2c38614ea5eb629251ec3e7cf52a1f4ef987cdabfcf4ae40d4` |
| `tools/crossseed_verify_v4.py` | `f6e3beec17791bf4420e35dc4fe5130c51758c7db4d74f38983bbbf52b1d070d` |
| `tools/test_crossseed_gate_v4.py` | `bcbf990f09520db3412b696713e402ff675c82fe5139cb0b5b947ef34a15978d` |
| `ssdd_dev_protocol/crossseed_orchestrator_v4.py` | `7caffb240080ccd791e797379cd0392002d6bbf61e002aa44a280e5daf45861a` |

v3 四个文件哈希复核后与上一轮记录一致，未被改动。

**相对 v3 的六处收紧：**

1. 代码身份改为「导入前钉住 `CODE_ROOT`、导入后断言」。实测钉住后可执行副本从
   editable 安装指向的 `DualTeacher/ssod` 变为 `DualTeacher_m3/ssod`，
   真实模型类源码解析为 `DualTeacher_m3/ssod/models/dual_teacher.py`。
   加载到预期根目录之外的副本 = 核验**失败**，不再只是「记录其哈希」。
   注：不再仅凭 `python -m torch.distributed.launch` 推断训练子进程加载了哪个副本。
2. 进程匹配按各进程自己的 `/proc/<pid>/cwd` 解析相对 `--work-dir`，并比对
   `--seed`，记录 pid 与启动时刻；信息读不到 = 无法判定 → 停止队列。
3. 退出码在训练返回后**立刻**落盘（`run_exit_status.json`），先于日志同步与核验；
   落盘前退出则保留「退出码未知」。
4. 交接时**总是重新核验**，不因存盘报告 `ok=true` 而复用；人工批准接受后、
   正式入表前各再核对一次产物是否变化。
5. 单实例锁（`crossseed_orchestrator.lock`，非阻塞排他 flock）。
6. 普通运行按已落盘退出码重启恢复，不再让队列死等。

**测试结果（合成/故障）：** 96 项检查，96 通过，0 失败。见交付包
`logs/fault_tests_v4.log`。覆盖 9 项行为表 (a)–(i)。

**真实权重核验（CPU，seed678 M0/M2 fold6 既有 32000 迭代 checkpoint）：**

| 用例 | 结果 |
|------|------|
| M0 fold6 seed678 | PASS，sha256 `e03857138f773470…` |
| M2 fold6 seed678 | PASS，sha256 `37d29bedc32adad0…`（与上一轮复现一致） |
| N1 M0 扮 M2 | 正确拒绝（exp_name + m2_enabled） |
| N2 M2 扮 M0 | 正确拒绝 |
| N3 错种子 123 | 正确拒绝（meta.seed + log seed） |
| N4 错 fold 7 | 正确拒绝（fold + load2_from） |
| N5 保留策略 10≠1 | 正确拒绝（max_keep_ckpts） |
| N6 在跑的 run #1（未完成） | 正确拒绝：`log last train iter is 7150, expected 32000` |

**编排器 dry-run：** 12 项计划全部正确；`seed123/m0/3/6` 为 `handover-pending`，
其余 11 项 `pending`；账本未改动（仍只有表头）。

**交付：** `~/dual_teacher_project/v4_bundle_20260919.tar.gz`
（sha256 `c7b1fe2965d433c8fbfd59811a2e0d11cfa8e1878330264f8ac010a4ecfa08a0`），
内含四个源文件、完整测试日志、真实权重核验 JSON、dry-run 日志与哈希清单。

**尚未独立复核：** 以上均为训练机上的执行结果；v4 尚未在真实排队的交接流程上
跑过端到端（当前运行未完成，`iter_32000.pth` 尚不存在）；
两份 `dual_teacher.py` 哈希相同只证明该文件内容相同，**尚未发现该模型文件的内容差异**，
但未对全部源码做全量比对。

**当前状态：** 训练 PID 2319534 `STAT=Rl` 正常推进（已跑 1h55m）；
v3 编排 PID 2442194 仍在 screen `m2crossseed` 中等待。

---

## run #1 完成与「代码来源」核查（2026-09-20）

**run #1 已训练完成。** `seed123/m0/3/6`，2026-09-19 16:57:11 → 2026-09-20 01:53:00，耗时 8h55m。
末行 `Iter [32000/32000] loss 1.0697 grad_norm 6.9812`，无 NaN/Inf，`iter_32000.pth` 已写出。

v3 编排器于 01:53:39 检测到训练结束，01:53:57 核验 **PASS**
（`sha256 5ba360d43e0196d6fbbfae2d8131e216190c62e7d90cdb0680b605781609c9b3`），
随后写出 `HANDOVER_RELEASE_REQUIRED.json` 并等待人工放行。
`request_id = 688d9ca397555b0cf6581900e87b5fdb44f5355993381edd759a27e685ef59e2`。
**尚未创建 `HANDOVER_RELEASED`，账本仍只有表头，11 项待跑。**

用 v4 核验器对该运行做独立交叉核验：同样 PASS，sha256 一致；
并确认加载的实现源码为 `DualTeacher_m3/ssod/models/dual_teacher.py`（`under_expected_root=true`）。

### 核查发现的代码来源问题（影响放行判断，如实记录）

实测（`easy-install.pth` 把 `/home/xcc/dual_teacher_project/DualTeacher` 放进 `sys.path`）：

| 角色 | 启动方式 | `import ssod` 解析到 |
|------|---------|---------------------|
| 训练 launcher | `python -m torch.distributed.launch`（cwd=`DualTeacher_m3`） | `DualTeacher_m3/ssod` |
| **训练 worker** | `python -u tools/train.py`（`sys.path[0]` = `DualTeacher_m3/tools`） | **`DualTeacher/ssod`** |
| v3 核验器 | 直接导入 | `DualTeacher_m3/ssod` |

`torch 1.7.0` 的 `launch.py` 只设置 `MASTER_ADDR/MASTER_PORT/WORLD_SIZE/OMP_NUM_THREADS/RANK/LOCAL_RANK`，
**不设置 `PYTHONPATH`**；`python -u tools/train.py` 也不会把仓库根目录放进 `sys.path`。
因此 **run #1 实际产出权重的 worker 加载的是 `DualTeacher/ssod`**，
而 v3 核验哈希的是 `DualTeacher_m3/ssod/models/dual_teacher.py` —— 核验的对象并非训练用的那份。

当前实际影响：对**整个 `ssod` 树**做 `diff -rq`（排除 `__pycache__`），
**内容不同的文件 0 个**；唯一差异是 `DualTeacher/ssod/models/` 下多一个
`dual_teacher.py.m2_backup_bb87fd7d`（开发期备份，不属于导入路径）。
即：**未发现任何源码内容差异**，但这是事后比对，不是运行时的加载证据。

**v4 的现状：** v4 只对**核验进程**做了代码根的钉住与断言，**没有**约束训练子进程的代码根，
因此单靠 v4 不能消除上述错配。切换编排前需要在训练子进程上显式固定代码根
（例如设 `PYTHONPATH=DualTeacher_m3`）并把预期代码根记入启动时 manifest。
**该项尚未修改，待确认后再动 v4 文件。**
