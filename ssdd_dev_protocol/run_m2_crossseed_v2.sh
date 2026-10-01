#!/bin/bash
# M2 跨种子复验 v2：训练 -> 自动核验 -> SHA256 入表 -> 立即启动下一次。
# 串行，无需人工确认。核验失败或训练非 0 退出即停队列并告警。
# 不做 AP 评估作为放行条件。
#
# 放行分两类：
#   (A) 由本脚本启动的 run：真实退出码必须为 0，且全部核验通过 -> 自动放行。
#   (B) 交接 run（非本脚本启动，真实退出码不可得）：核验通过后写
#       HANDOVER_RELEASE_REQUIRED.json，等待人工放行标记 HANDOVER_RELEASED
#       才继续；记录 exit_code=null / exit_status=unknown_due_to_handover，
#       不伪造退出码 0。仅为 run #1 设计，一次性。
#
# 幂等跳过：不仅要求 VERIFIED.json 存在，还要求其 seed/ver/fold/run_dir/iters
# 与当前目标一致，且其中的 sha256 与磁盘上 iter_32000.pth 的实际哈希相同
# （防止标记仍在而权重已缺失或被替换）。
set -uo pipefail
cd /home/xcc/dual_teacher_project/DualTeacher_m3
PY=/home/xcc/anaconda3/envs/dt/bin/python
MAXITERS=32000
RUNS_TSV=ssdd_dev_protocol/m2_crossseed_runs.tsv
V2LOG=ssdd_dev_protocol/run_m2_crossseed_v2.log
MIN_DISK_GB=10

log() { echo "[$(date '+%F %T')] $*" | tee -a "$V2LOG"; }

cfg_for() {
  case "$1" in
    m0) echo configs/reproduce/phase3_dual_teacher_ssdd_dev.py ;;
    m2) echo configs/reproduce/phase3_dual_teacher_ssdd_dev_m2.py ;;
    *)  echo "bad ver: $1" >&2; return 1 ;;
  esac
}

training_running() { pgrep -f "tools/train.py" >/dev/null 2>&1; }

# Wait until the in-flight run in $1 has finished. 0 if it produced
# iter_32000.pth, 1 if training is gone without a checkpoint.
wait_for_inflight() {
  local run_dir=$1
  while true; do
    if [ -f "$run_dir/iter_32000.pth" ] && ! training_running; then
      return 0
    fi
    if ! training_running && [ ! -f "$run_dir/iter_32000.pth" ]; then
      sleep 30
      if ! training_running && [ ! -f "$run_dir/iter_32000.pth" ]; then
        return 1
      fi
    fi
    sleep 60
  done
}

# 0 only if a VERIFIED.json exists AND matches this target AND its sha256 is
# still the hash of the checkpoint on disk.
verified_ok() {
  local run_dir=$1 seed=$2 ver=$3 fold=$4
  [ -f "$run_dir/VERIFIED.json" ] || return 1
  [ -f "$run_dir/iter_32000.pth" ] || return 1
  $PY - "$run_dir" "$seed" "$ver" "$fold" "$MAXITERS" <<'EOF' >/dev/null 2>&1
import hashlib, json, os, sys
run_dir, seed, ver, fold, iters = sys.argv[1:6]
d = json.load(open(os.path.join(run_dir, "VERIFIED.json")))
ck = os.path.join(run_dir, "iter_32000.pth")
h = hashlib.sha256()
with open(ck, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
ok = (os.path.abspath(d.get("run_dir", "")) == os.path.abspath(run_dir)
      and int(d.get("seed", -1)) == int(seed)
      and str(d.get("ver")) == ver
      and int(d.get("fold", -1)) == int(fold)
      and int(d.get("iters", -1)) == int(iters)
      and d.get("sha256") == h.hexdigest())
sys.exit(0 if ok else 1)
EOF
}

if [ ! -s "$RUNS_TSV" ]; then
  echo -e "seed\tver\tfold\trun_dir\tsha256\titers\texit_code\texit_status\tverified_at" > "$RUNS_TSV"
fi

for seed in 123 456; do
  for fold in 6 7 8; do
    for ver in m0 m2; do
      cfg=$(cfg_for "$ver") || exit 1
      run_dir="work_dirs/m2_crossseed/seed${seed}/${ver}/3/${fold}"

      if verified_ok "$run_dir" "$seed" "$ver" "$fold"; then
        log "skip (verified marker matches current checkpoint): $run_dir"
        continue
      fi

      handover=0
      exit_code=0
      exit_status="ok"

      if [ -e "$run_dir" ]; then
        # 非本脚本启动的 run：真实退出码不可得，一律走人工放行。
        handover=1
        exit_status="unknown_due_to_handover"
        if [ ! -f "$run_dir/iter_32000.pth" ] || training_running; then
          log "handover: waiting for in-flight $run_dir"
          if ! wait_for_inflight "$run_dir"; then
            log "FAIL: in-flight $run_dir ended without iter_32000.pth; stopping queue"
            exit 1
          fi
        fi
        log "handover: $run_dir finished; verifying (exit code not available)"
      else
        log "=== START seed${seed} ${ver} fold${fold} $(date '+%F %T') ==="
        $PY -m torch.distributed.launch --nproc_per_node=1 \
          tools/train.py "$cfg" \
          --launcher pytorch \
          --seed "$seed" \
          --no-validate \
          --work-dir "$run_dir" \
          --cfg-options \
          fold="$fold" percent=3 \
          auto_resume=False \
          runner.max_iters=$MAXITERS \
          checkpoint_config.max_keep_ckpts=1
        exit_code=$?
        log "=== END seed${seed} ${ver} fold${fold} rc=${exit_code} $(date '+%F %T') ==="
        if [ "$exit_code" -ne 0 ]; then
          log "FAIL: training exit code ${exit_code} for $run_dir; stopping queue"
          exit 1
        fi
      fi

      # ---- release gate: verify before starting the next run ----
      if ! $PY tools/verify_m2_crossseed_ckpt.py "$cfg" "$run_dir/iter_32000.pth" \
             --run-dir "$run_dir" --seed "$seed" --expected-iters $MAXITERS \
             --cfg-options fold="$fold" percent=3 \
             --out "$run_dir/verify_result.json" >> "$V2LOG" 2>&1; then
        log "FAIL: verification failed for $run_dir (see $run_dir/verify_result.json); stopping queue"
        exit 1
      fi
      sha=$($PY -c "import json;print(json.load(open('$run_dir/verify_result.json'))['checks']['sha256'])")
      log "VERIFY OK $run_dir sha256=$sha"

      if [ "$handover" -eq 1 ]; then
        cat > "$run_dir/HANDOVER_RELEASE_REQUIRED.json" <<EOF
{
  "run_dir": "$run_dir",
  "seed": $seed,
  "ver": "$ver",
  "fold": $fold,
  "sha256": "$sha",
  "exit_code": null,
  "exit_status": "unknown_due_to_handover",
  "verification": "passed",
  "reason": "run was started by the previous (now terminated) launcher; its real exit code cannot be captured",
  "release_action": "create the marker file $run_dir/HANDOVER_RELEASED to continue the queue"
}
EOF
        log "HANDOVER: verification PASSED for $run_dir; awaiting one-time manual release."
        log "HANDOVER: create '$run_dir/HANDOVER_RELEASED' to continue the queue."
        while [ ! -f "$run_dir/HANDOVER_RELEASED" ]; do sleep 60; done
        log "HANDOVER: release marker found; continuing queue."
      fi

      printf '{"run_dir":"%s","seed":%d,"ver":"%s","fold":%d,"sha256":"%s","iters":%d,"exit_code":%s,"exit_status":"%s","manual_release":%s,"verified_at":"%s"}\n' \
        "$run_dir" "$seed" "$ver" "$fold" "$sha" "$MAXITERS" \
        "$([ "$handover" -eq 1 ] && echo null || echo "$exit_code")" \
        "$exit_status" \
        "$([ "$handover" -eq 1 ] && echo true || echo false)" \
        "$(date '+%F %T')" \
        > "$run_dir/VERIFIED.json"
      if [ "$handover" -eq 1 ]; then
        echo -e "${seed}\t${ver}\t${fold}\t${run_dir}\t${sha}\t${MAXITERS}\tnull\t${exit_status}\t$(date '+%F %T')" >> "$RUNS_TSV"
      else
        echo -e "${seed}\t${ver}\t${fold}\t${run_dir}\t${sha}\t${MAXITERS}\t${exit_code}\tok\t$(date '+%F %T')" >> "$RUNS_TSV"
      fi
      rm -f "$run_dir/HANDOVER_RELEASE_REQUIRED.json"

      avail=$(df -BG --output=avail /home/xcc | tail -1 | tr -dc '0-9')
      log "disk avail ${avail}GB (min ${MIN_DISK_GB}GB)"
      if [ "$avail" -lt "$MIN_DISK_GB" ]; then
        log "FAIL: disk headroom ${avail}GB < ${MIN_DISK_GB}GB; stopping queue"
        exit 1
      fi
    done
  done
done

log "ALL M2 CROSSSEED DONE $(date '+%F %T')"
