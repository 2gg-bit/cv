# Dual Teacher: A Semi-Supervised Co-Training Framework for Cross-Domain Ship Detection


## Introduction
This is the reserch code of the IEEE Transactions on Geoscience and Remote Sensing 2023 paper.

X. Zheng, H. Cui, C. Xu and X. Lu, "Dual Teacher: A Semi-Supervised Co-Training Framework for Cross-Domain Ship Detection," IEEE Transactions Geoscience and Remote Sensing, 2023.

In this code, we explored the Semi-Supervised Cross-Domain Ship Detection (SCSD) task to improve the cross-domain ship detection performance with a few labeled SAR images. We proposed Dual Teacher framework to integrate cross-domain object detection and semi-supervised object detection for different knowledge fusion.

## Usage

### Requirements
- `Ubuntu 20.04`
- `Anaconda3` with `python=3.6`
- `Pytorch=1.7.0`
- `mmdetection=2.16.0+fe46ffe`
- `mmcv=1.3.9`
- `wandb=0.10.31`


### Installation
```
make install
```

### Data 
- Download DIOR, HRSID and SSDD datasets and put them as follows:
- Execute the following command to generate data set splits:
```shell script
# YOUR_DATA/
#   dior/
#     dior_annotations.json    # only ship instances
#     images/
#   hrsid/
#     annotations/
#     images/
#   ssdd/
#     annotations/
#     JPEGImages/
#   dior_hrsid/
#     annotations/             # labeled optical images and few labeled SAR images
#     images/
ln -s ${YOUR_DATA} data
bash tools/dataset/semi_hrsid.sh
bash tools/dataset/semi_ssdd.sh
```
- ADD HRSIDDataset to MMDetection, similar to COCODataset

### Training
```shell script
# num_SAR_images: number of labeled SAR images for training
# num_gpus: number of gpus for training
bash tools/dist_train_ship_pretrain.sh dior 0 100 ${num_gpus}
for fold in 1,2,3,4,5;
do
    bash tools/dist_train_ship_pretrain.sh dior_hrsid ${fold} ${num_SAR_images} ${num_gpus}
    bash tools/dist_train_dual_teacher_partially_hrsid.sh semi ${fold} ${num_SAR_images} ${num_gpus}
done 
```
### Evaluation
```shell script
python tools/test.py <config_file_path> <model_file_path> --eval bbox --work-dir <save_dir>
```

### Corrected SSDD baseline: initialization and NMS

The corrected fresh-run path explicitly loads Phase 1 into teacher1/student1
and Phase 2 into teacher2/student2 **after** generic model initialization and
**before** the runner's first step / EMA hook. Every parameter and buffer is
checked for matching keys, shapes, finite values and equality after loading.
Missing/incompatible checkpoints stop training. A successful startup prints
four `[DualTeacher init] ... verified ...` messages followed by:

```text
[DualTeacher init] PASS: T1=S1, T2=S2, T1!=T2; fusion=NMS, fusion_iou=0
```

Pseudo-label fusion now follows the released author code's ordinary NMS,
including **fusion IoU=0** and empty-teacher passthrough. This is separate from
the detector's own NMS thresholds. The previous consensus OR score boost and
single-teacher rescaling have been removed. The learning rate, 32000 iterations,
loss weights, data splits and EMA schedule of the single-GPU reproduction config
are otherwise unchanged; this is not a claim of identical four-GPU optimization.

On the training machine, first check the real checkpoints without a dataset,
GPU training or optimizer (repeat for folds 6, 7 and 8):

```shell
python tools/check_dual_teacher_init.py configs/reproduce/phase3_dual_teacher_ssdd.py \
    --cfg-options fold=6 percent=3
```

After the checks pass, launch a **new** Phase 3 run, retaining the same labeled
images. For example (choose and record the training RNG seed deliberately;
the fold number only selects the labeled-data split):

```shell
python -m torch.distributed.launch --nproc_per_node=1 \
    tools/train.py configs/reproduce/phase3_dual_teacher_ssdd.py \
    --launcher pytorch --seed 678 --cfg-options fold=6 percent=3
```

Outputs go to `work_dirs/phase3_dual_teacher_baseline_nms/3/6`, not the old
`phase3_dual_teacher` directory; automatic resume is disabled. **Do not resume
the old consensus/uninitialized Phase 3 checkpoints as a corrected baseline.**
Existing Phase 1/2 checkpoints and all old logs should be kept unchanged.
Resume a corrected run only with an explicit `--resume-from` pointing to its
full four-branch checkpoint. Full checkpoint restore/inference does not need
the Phase 1/2 files and will not substitute their weights.

CPU regression tests use small synthetic checkpoints and real PyTorch/NMS,
with MMDetection construction replaced by lightweight fixtures:

```shell
python -m pytest -q tests/test_dual_teacher_baseline.py
```

These tests require PyTorch, NumPy, Numba and pytest. They do not replace the
real-checkpoint startup check above or a full CUDA training experiment.

## Cite
```
@article{zheng2023dual,
  author={Zheng, Xiangtao and Cui, Haowen and Xu, Chujie and Lu, Xiaoqiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Dual Teacher: A Semisupervised Cotraining Framework for Cross-Domain Ship Detection}, 
  year={2023},
  volume={61},
  number={},
  pages={1-12},
  doi={10.1109/TGRS.2023.3287863}}
```

## Acknowledgement
A large part of the codes are borrowed from [SoftTeacher](https://github.com/microsoft/SoftTeacher). Thanks for the excellent work!
