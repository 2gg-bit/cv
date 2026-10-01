# SSDD 开发集协议（DIOR+SSDD 可信对照）

目标：继续用 DIOR+SSDD，从 SSDD 原训练池建立可信开发集，判断 Soft-NMS 是否值得保留，
再决定新增哪一种单变量改动。**暂不添加网络部件，尚未启动长训。**

## 已交付四项（本阶段目标）

| 交付物 | 文件 |
|--------|------|
| 1. 实验协议 | `protocol.json` |
| 2. 开发划分清单 | `data/split_manifest.json`（+ `data/dev.json`、`data/train_pool.json`、`data/dev_image_ids.json`、`data/leakage_check.json`） |
| 3. 数据隔离检查 | `data/isolation_check.json` |
| 4. 三组新配置 | `configs/reproduce/phase2_pretrain_optical_sar_dev.py`、`configs/reproduce/phase3_dual_teacher_ssdd_dev.py`（fold 6/7/8 参数化） |

## 开发划分要点

- 源：SSDD `train.json`（928 图 / 2041 框）。
- 开发集：**186 图（≈20%）**，从非监督池采样，`numpy RandomState(678)`。
- 排除：三组原有图像选取的 **9 张监督图**（fold 6/7/8 各 3 张）全部留在训练池。
- 去重：发现 **14 组精确重复（29 图）**，按内容 MD5 原子成组，**无一组被拆分到开发/训练两侧**。
- 场景：标注中无显式场景元数据；仅做精确重复分组，**不声称**图像编号级场景隔离（记录为限制）。
- 泄漏核验：开发图 ∩ 监督图 = ∅、开发图 ∩ 测试集 = ∅、开发图全部来自训练池、监督图全部保留、
  开发图不在 Phase2 混合 / Phase3 sup2 / 新无标签 任何训练入口（`isolation_check.json` 全零）。

## 配置改动（仅限数据隔离 + 输出目录 + 初始化路径，推理不变）

- `phase2_..._dev.py`：仅 `work_dir → work_dirs/dev_ssdd/phase2_pretrain_optical_sar/...`（数据入口与推理不变）。
- `phase3_..._dev.py`：`unsup` 改用 `ssdd_dev_protocol/data/instances_train2017.{fold}@{percent}-unlabeled-dev.json`；
  `load2_from` 指向 dev_ssdd 新 Phase2；`work_dir → work_dirs/dev_ssdd/phase3_dual_teacher/...`。
- 训练种子 678、32000 iter、`iter_32000.pth` 最终权重、teacher2 推理、A=Hard-NMS / B=Linear Soft-NMS 均见 `protocol.json`。

## 必须披露

本开发集额外使用了 186 张 SSDD 训练图标注。只能说“训练监督为三张 SAR 图像”，
**不能说整个方法开发仅使用三张标注图**。

## 权重复用方案（阶段一已确定，见 weight_reuse.json）

- **Phase1：复用**。纯 DIOR（`work_dirs/phase1_pretrain_optical/100/1/iter_8000.pth`，SHA256 已记录），不含 SSDD。
- **Phase2：重训**。旧 Phase2 数据（2706 DIOR + 3 监督 SSDD）未见开发图，但旧 Phase2 训练于 seed=678 协议之前、种子未固定，不完全满足新协议 → 用 `phase2_..._dev.py` 以 seed=678 重训。

## 下一步（阶段二，尚未执行）

1. Phase2 重训（三组，seed=678）→ 严格初始化检查（T1=S1、T2=S2、T1!=T2，未启用 M1）。
2. Phase3 短训冒烟（第一组 100 iter，独立目录）→ 验收后三组正式训练 32000 iter。
3. 开发集固定 A/B 评估；**不根据第一组结果决定另外两组**，不搜索阈值。
