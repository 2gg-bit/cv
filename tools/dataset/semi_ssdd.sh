#!/usr/bin/env bash
set -x
data_root="${SSDD_DATA_DIR:-data/ssdd}"
OFFSET=$RANDOM
for percent in 1 3 5 10; do
    for fold in 1 2 3 4 5; do
        python tools/dataset/semi_ssdd.py --percent ${percent} --seed ${fold} --data-dir "${data_root}" --seed-offset ${OFFSET}
    done
done
