#!/usr/bin/env bash
# 三折新 B0 与 PG-both 离线误差分析：正式运行 + 确定性逐项比较。
#
# 用法：bash run_analysis.sh
#
# 约定：
#   - run_001 为正式产物；run_002 仅用于确定性复跑比较；两者都必须不存在或为空。
#   - 任一 run 非零退出即停止，不写"分析完成"状态，已产生的失败材料保留。
#   - 不复跑推理、不训练、不改动任何输入文件。

set -euo pipefail

PY=/home/xcc/anaconda3/envs/dt/bin/python
ROOT=/home/xcc/dual_teacher_project/DualTeacher_ablation/ablation_configs/error_analysis_b0_pgboth
CODE="$ROOT/code/error_analysis.py"
RUN1="$ROOT/results/run_001"
RUN2="$ROOT/results/run_002"

for d in "$RUN1" "$RUN2"; do
  if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
    echo "拒绝覆盖：$d 已存在且非空" >&2
    exit 1
  fi
done

INPUTS=(
  "$ROOT/../fold6_seed678/b0_evaluation/predictions.bbox.json"
  "$ROOT/../fold6_seed678/pg_both_evaluation/predictions.bbox.json"
  "$ROOT/../fold7_seed678/b0_evaluation/predictions.bbox.json"
  "$ROOT/../fold7_seed678/pg_both_evaluation/predictions.bbox.json"
  "$ROOT/../fold8_seed678/b0_evaluation/predictions.bbox.json"
  "$ROOT/../fold8_seed678/pg_both_evaluation/predictions.bbox.json"
  "$ROOT/../fold6_seed678/b0_evaluation/test.json"
)

hash_inputs() {
  for f in "${INPUTS[@]}"; do sha256sum "$f"; done
}

BEFORE=$(hash_inputs)

echo "=== 依赖自审（AST） ==="
"$PY" - "$CODE" <<'EOF'
import ast, sys
FORBIDDEN = {"ssod", "mmdet", "mmcv", "mmcv_full", "torch"}
path = sys.argv[1]
tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
hits, mods = [], set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for a in node.names:
            mods.add(a.name.split(".")[0])
            if a.name.split(".")[0] in FORBIDDEN:
                hits.append((node.lineno, a.name))
    elif isinstance(node, ast.ImportFrom):
        top = (node.module or "").split(".")[0]
        mods.add(top)
        if top in FORBIDDEN:
            hits.append((node.lineno, node.module))
print("顶层模块:", ", ".join(sorted(m for m in mods if m)))
assert not hits, "禁止的导入: %r" % hits
print("OK：无 ssod/mmdet/mmcv/torch 导入")
EOF

echo
echo "=== run_001（正式） ==="
"$PY" "$CODE" run --out-dir "$RUN1"

echo
echo "=== run_002（确定性复跑） ==="
"$PY" "$CODE" run --out-dir "$RUN2"

echo
echo "=== 确定性逐项比较（排除 run_metadata.json） ==="
"$PY" - "$RUN1" "$RUN2" <<'EOF'
import hashlib, json, os, sys

run1, run2 = sys.argv[1], sys.argv[2]

def collect(root):
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn == "run_metadata.json":
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root)
            with open(p, "rb") as f:
                out[rel] = hashlib.sha256(f.read()).hexdigest()
    return out

a, b = collect(run1), collect(run2)
miss_a = sorted(set(a) - set(b))
miss_b = sorted(set(b) - set(a))
diff = sorted(k for k in set(a) & set(b) if a[k] != b[k])

print("比较文件数:", len(a))
print("仅 run_001 有:", miss_a or "无")
print("仅 run_002 有:", miss_b or "无")
print("内容不同:", diff or "无")
assert not miss_a and not miss_b and not diff, "确定性比较失败"

# manifest 中必须一致的字段
m1 = json.load(open(os.path.join(run1, "manifest.json"), encoding="utf-8"))
m2 = json.load(open(os.path.join(run2, "manifest.json"), encoding="utf-8"))
same_keys = ["analysis_nature", "mechanism_conclusion", "analysis_plan", "analysis_plan_v1",
             "script", "environment", "frozen_test_sha256",
             "num_images", "num_gt_noncrowd", "constants", "outputs"]
for k in same_keys:
    assert m1[k] == m2[k], "manifest 字段不一致: %s" % k
print("manifest 一致字段:", ", ".join(same_keys))

s1 = json.load(open(os.path.join(run1, "summary.json"), encoding="utf-8"))
s2 = json.load(open(os.path.join(run2, "summary.json"), encoding="utf-8"))
assert s1 == s2 and s1.get("status") == "complete"
print("summary.status:", s1["status"])
print("确定性比较：全部一致")
EOF

echo
echo "=== 输入文件哈希前后不变 ==="
AFTER=$(hash_inputs)
if [ "$BEFORE" = "$AFTER" ]; then
  echo "OK：${#INPUTS[@]} 个输入文件哈希前后一致"
else
  echo "输入文件被改动！" >&2
  exit 1
fi

echo
echo "全部通过。正式产物：$RUN1"
