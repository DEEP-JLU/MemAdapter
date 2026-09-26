#!/usr/bin/env bash
# Lane launcher for the generation and judging stages of the external benchmarks.
#
# Separate from run_external_retrieval.sh because the two stages are bound by
# different resources. Retrieval is RAM-bound: every shard process holds a store
# client and its own embedder handle, so its width is set by available memory.
# Generation and judging are network-bound: a process holds no store and no embedder,
# and its width is `--workers` in-flight requests. One cap for both would cap the
# wrong thing on one of them -- so retrieval keeps its own launcher (and its own RAM
# gate) and this one owns everything after it.
#
# A cell is one (task, system). Its job runs, in order: every repeat's baseline arm,
# that repeat's MemAdapter arm, then both arms' judgements. Runs are ordered that way
# so a cell produces reportable numbers the moment its last repeat is judged, rather
# than at the end of the lane.
#
# PIPELINING, and why the guard is what it is: a cell starts as soon as its own
# merged retrieval file exists, not when the lane finishes. MemTrapBench is the case
# that matters -- its lane merges (task, system) groups in the order the shard pool
# reaches them, so the first groups generate for hours while the last ones are still
# building stores. Two properties make that safe rather than lucky:
#
#   * build_retrieval.py --merge refuses to write the final file until every expected
#     sample is present exactly once, so *the file existing means the group is whole*.
#     The one exception is --allow-partial, which is a pilot-only flag; passing it here
#     turns that implication off, so this script then checks the row count itself.
#   * the arms re-derive their own protocol record from the file they are handed
#     (run_config.config_for_external_arm) and stop if it disagrees with the one
#     already in the cell directory. A cell generated from a partial file therefore
#     fails loudly on its first row instead of producing a short result set.
#
# Usage:
#   run_external_arms.sh --lane persist --workers 32            # usual full run
#   run_external_arms.sh --lane both --jobs 4 --workers 32      # both lanes, one call
#   run_external_arms.sh --lane persist --no-retrieval          # retrieval already frozen
#   run_external_arms.sh --lane persist --tasks persist_cross_domain --systems AMEM \
#     --no-retrieval --allow-partial --limit 2 --jobs 1 --workers 8   # pilot
#   run_external_arms.sh --lane both --dry-run                  # the call matrix
set -Eeuo pipefail

# See run_external_retrieval.sh.  A direct PowerShell -> Git Bash invocation
# otherwise inherits only the Windows PATH and cannot find dirname/tr/cut.
export PATH="/usr/bin:/bin:${PATH}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
LOG_ROOT="$ROOT/external_results/logs/arms"

# Every path that a *shell variable* carries is computed relative to ROOT, and this
# script runs from ROOT so the runners resolve them. That is not a style choice: the
# repository path contains CJK characters, and Python printing an absolute path to a
# pipe encodes it with the locale code page (cp936 here) while bash holds its own
# UTF-8. The shell then compares two different byte sequences for one file and finds
# nothing -- which is how a launcher that had already found the merged retrieval file
# reports "needs retrieval" for every cell. Relative paths keep the whole pipeline
# ASCII, and PYTHONIOENCODING keeps any remaining non-ASCII output unambiguous in the
# logs.
cd "$ROOT"
export PYTHONIOENCODING=utf-8

LANE="both"
SYSTEMS="AMEM,Mem0,naiveRAG"
TASKS=""
JOBS="4"
RETRIEVAL_JOBS="6"
WORKERS="32"
MODEL="DeepSeek"
LIMIT=""
RETRIEVAL="1"
SHARDS=""
ALLOW_PARTIAL="0"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lane)          LANE="$2"; shift 2 ;;
    --systems)       SYSTEMS="$2"; shift 2 ;;
    --tasks)         TASKS="$2"; shift 2 ;;
    --jobs)          JOBS="$2"; shift 2 ;;
    --retrieval-jobs) RETRIEVAL_JOBS="$2"; shift 2 ;;
    --workers)       WORKERS="$2"; shift 2 ;;
    --model)         MODEL="$2"; shift 2 ;;
    --limit)         LIMIT="$2"; shift 2 ;;
    --shards)        SHARDS="$2"; shift 2 ;;
    --min-free-mib)  MIN_FREE_MIB="$2"; shift 2 ;;
    --no-retrieval)  RETRIEVAL="0"; shift ;;
    --allow-partial) ALLOW_PARTIAL="1"; shift ;;
    --dry-run)       DRY_RUN="1"; shift ;;
    -h|--help)       sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
MIN_FREE_MIB="${MIN_FREE_MIB:-1536}"

[[ "$LANE" =~ ^(memtrap|persist|both)$ ]] || { echo "--lane must be memtrap|persist|both" >&2; exit 2; }

# One line per cell: task dataset subset system k retrieval_file. The retrieval path
# and the repeat count both come from the registry rather than being spelled here, so
# a cell this script dispatches cannot disagree with the one the runners resolve.
list_cells() {
  "$PYTHON" - "$ROOT" "$LANE" "$TASKS" "$SYSTEMS" <<'PY' | tr -d '\r'
import os
import sys
sys.path.insert(0, sys.argv[1])
from external_benchmarks import build_retrieval, dataset_adapters, paths, task_registry
lane, explicit, systems = sys.argv[2], sys.argv[3], [s for s in sys.argv[4].split(",") if s]
dataset = None if lane == "both" else ("memtrapbench" if lane == "memtrap" else "persistbench")
tasks = [t.strip() for t in explicit.split(",") if t.strip()] or list(dataset_adapters.iter_tasks(dataset))
for task in tasks:
    spec = task_registry.spec(task)
    for system in systems:
        path = paths.retrieval_dir(spec.dataset, spec.subset, system) / build_retrieval.output_name(spec)
        print(task, spec.dataset, spec.subset, system, spec.k_generations,
              os.path.relpath(path, sys.argv[1]).replace(os.sep, "/"))
PY
}

cell_rows() {
  # Expected rows for a cell's retrieval file, printed as "-" if the file is absent.
  "$PYTHON" - "$ROOT" "$1" "$2" <<'PY' | tr -d '\r'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
path = Path(sys.argv[3])
print(sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.is_file() else "-")
PY
}

expected_rows() {
  "$PYTHON" - "$ROOT" "$1" <<'PY' | tr -d '\r'
import sys
sys.path.insert(0, sys.argv[1])
from external_benchmarks import dataset_adapters
print(dataset_adapters.sample_count(sys.argv[2]))
PY
}

arm_dir() {
  "$PYTHON" - "$ROOT" "$1" "$2" "$3" "$4" "$5" <<'PY' | tr -d '\r'
import os, sys
sys.path.insert(0, sys.argv[1])
from external_benchmarks import paths
path = paths.arm_dir(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]))
print(os.path.relpath(path, sys.argv[1]).replace(os.sep, "/"))
PY
}

# Every stage runs under with_protocol_env: the twelve pins (endpoint, model id,
# temperature, max_tokens, thinking) must be in the environment *before* the first
# ModelClient exists, because ModelClient reads its sampling settings at construction
# and offers no per-call temperature. Running the pins by hand once and forgetting
# them later is exactly how two arms end up on different settings.
run_stage() {
  local stage="$1" cell="$2" arm="$3" run="$4" log_file="$5"
  local task="${cell%%|*}"; local dataset; local subset; local system
  dataset="$(cut -d'|' -f2 <<<"$cell")"; subset="$(cut -d'|' -f3 <<<"$cell")"; system="$(cut -d'|' -f4 <<<"$cell")"
  local file; file="$(cut -d'|' -f5 <<<"$cell")"
  local extra=()
  [[ -n "$LIMIT" ]] && extra+=(--limit "$LIMIT")

  case "$stage" in
    baseline)
      PYTHONPATH="." "$PYTHON" -u -m external_benchmarks.with_protocol_env -- \
        "$PYTHON" -u -m external_benchmarks.run_external_baseline \
        --retrieval-file "$file" --system "$system" --model "$MODEL" \
        --run-index "$run" --workers "$WORKERS" --continue-on-error "${extra[@]}" \
        >>"$log_file" 2>&1
      ;;
    memadapter)
      PYTHONPATH="." "$PYTHON" -u -m external_benchmarks.with_protocol_env -- \
        "$PYTHON" -u "$ROOT/MemAdapter/run_memadapter.py" generate \
        --retrieval-file "$file" --memory-system "$system" --model "$MODEL" \
        --output-dir "$(arm_dir "$dataset" "$subset" "$system" memadapter "$run")" \
        --workers "$WORKERS" --continue-on-error \
        --allow-fewer-than-top-10 "${extra[@]}" \
        >>"$log_file" 2>&1
      ;;
    judge)
      PYTHONPATH="." "$PYTHON" -u -m external_benchmarks.with_protocol_env -- \
        "$PYTHON" -u -m external_benchmarks.run_external_judge \
        --retrieval-file "$file" --system "$system" --arm "$arm" \
        --run-index "$run" --model "$MODEL" --workers "$WORKERS" \
        --continue-on-error "${extra[@]}" \
        >>"$log_file" 2>&1
      ;;
    *) echo "unknown stage: $stage" >&2; return 2 ;;
  esac
}

run_cell() {
  cell="$1"
  task="${cell%%|*}"; dataset="$(cut -d'|' -f2 <<<"$cell")"
  subset="$(cut -d'|' -f3 <<<"$cell")"; system="$(cut -d'|' -f4 <<<"$cell")"
  k="$(cut -d'|' -f6 <<<"$cell")"
  dir="$LOG_ROOT/$dataset/$subset/$system"
  mkdir -p "$dir"
  for ((run = 0; run < k; run++)); do
    for stage in baseline memadapter; do
      run_stage "$stage" "$cell" "" "$run" "$dir/${stage}_run${run}.log" || return 1
    done
    for arm in baseline memadapter; do
      run_stage judge "$cell" "$arm" "$run" "$dir/judge_${arm}_run${run}.log" || return 1
    done
  done
  return 0
}

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'lane=%s systems=%s jobs=%s workers=%s model=%s limit=%s retrieval=%s\n' \
    "$LANE" "$SYSTEMS" "$JOBS" "$WORKERS" "$MODEL" "${LIMIT:-<all>}" "$RETRIEVAL"
  printf '\n%-34s %-10s %5s %10s %s\n' cell system k rows ready
  while read -r task dataset subset system k file; do
    [[ -n "$task" ]] || continue
    rows="$(cell_rows "$task" "$file")"
    printf '%-34s %-10s %5s %10s %s\n' "$task" "$system" "$k" "$rows" \
      "$([[ "$rows" == "-" ]] && echo 'needs retrieval' || echo yes)"
  done < <(list_cells)
  exit 0
fi

LANE_DIR="$LOG_ROOT/_lane_$LANE"
mkdir -p "$LANE_DIR"
LANE_LOG="$LANE_DIR/lane.log"
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LANE_LOG"; }

RETRIEVAL_PID=""
if [[ "$RETRIEVAL" == "1" ]]; then
  # The retrieval lane keeps its own RAM gate and its own retry budget; the only thing
  # this script has to know about it is whether it is still running and whether a
  # given group's merged file has appeared.
  retr_args=(--lane "$LANE" --systems "$SYSTEMS" --jobs "$RETRIEVAL_JOBS" \
             --min-free-mib "$MIN_FREE_MIB")
  [[ -n "$TASKS" ]] && retr_args+=(--tasks "$TASKS")
  [[ -n "$SHARDS" ]] && retr_args+=(--shards "$SHARDS")
  [[ "$ALLOW_PARTIAL" == "1" ]] && retr_args+=(--allow-partial)
  log "starting retrieval lane: ${retr_args[*]}"
  ( "$ROOT/external_benchmarks/launcher/run_external_retrieval.sh" "${retr_args[@]}" \
      >>"$LANE_DIR/retrieval.log" 2>&1 ) &
  RETRIEVAL_PID=$!
fi

declare -A LIVE=()      # pid -> cell
declare -A DISPATCHED=() # cell -> 1
declare -A FAILED=()

dispatch_ready() {
  local dispatched=0
  while read -r task dataset subset system k file; do
    [[ -n "$task" ]] || continue
    local cell="$task|$dataset|$subset|$system|$file|$k"
    [[ -n "${DISPATCHED[$cell]:-}" ]] && continue
    # Only a complete merged file starts a cell: the presence of the file is the
    # completeness signal (see the header), except under --allow-partial, where the
    # row count has to be checked instead.
    [[ -f "$file" ]] || continue
    if [[ "$ALLOW_PARTIAL" == "1" ]]; then
      local rows want
      rows="$(cell_rows "$task" "$file")"
      want="$(expected_rows "$task")"
      # --allow-partial writes whatever the finished shards produced, so the file
      # existing no longer implies the group is whole. A limited pilot is expected to
      # be short, so it only has to be non-empty there; otherwise the row count must
      # be the dataset's, or the cell would generate from a truncated memory set.
      if [[ -n "$LIMIT" ]]; then
        (( rows > 0 )) || continue
      else
        (( rows == want )) || continue
      fi
    fi

    while :; do
      live=0
      for pid in "${!LIVE[@]}"; do
        kill -0 "$pid" 2>/dev/null && live=$((live + 1)) || unset "LIVE[$pid]"
      done
      (( live < JOBS )) && break
      sleep 5
    done

    ( run_cell "$cell" ) >>"$LANE_DIR/cells.log" 2>&1 &
    LIVE[$!]="$cell"
    DISPATCHED[$cell]=1
    dispatched=$((dispatched + 1))
    log "dispatched $task/$system"
  done < <(list_cells)
  return 0
}

log "lane=$LANE started; cells=$JOBS jobs=$JOBS workers=$WORKERS model=$MODEL"
while :; do
  dispatch_ready
  live=0
  for pid in "${!LIVE[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      live=$((live + 1))
    else
      wait "$pid" || FAILED["${LIVE[$pid]}"]=1
      unset "LIVE[$pid]"
    fi
  done
  if (( live == 0 )); then
    if [[ -n "$RETRIEVAL_PID" ]] && kill -0 "$RETRIEVAL_PID" 2>/dev/null; then
      sleep 20
      continue
    fi
    remaining=0
    while read -r task dataset subset system k file; do
      [[ -n "$task" ]] || continue
      cell="$task|$dataset|$subset|$system|$file|$k"
      [[ -n "${DISPATCHED[$cell]:-}" ]] || remaining=$((remaining + 1))
    done < <(list_cells)
    (( remaining == 0 )) && break
    # No cell is live and the retrieval lane has exited, yet cells remain undispatched:
    # their group never merged. Wait once more, then say so plainly rather than
    # spinning -- the retrieval log names the shard that failed.
    log "$remaining cell(s) still have no merged retrieval file; waiting 60s more"
    sleep 60
    dispatch_ready
    pending=0
    while read -r task dataset subset system k file; do
      [[ -n "$task" ]] || continue
      cell="$task|$dataset|$subset|$system|$file|$k"
      [[ -n "${DISPATCHED[$cell]:-}" ]] || pending=$((pending + 1))
    done < <(list_cells)
    (( pending == 0 )) && break
    log "giving up on $pending cell(s) with no merged retrieval file: see $LANE_DIR/retrieval.log"
    break
  fi
  sleep 15
done

# A cell whose group never merged is a failure of the lane, not an empty result set:
# it is the difference between "this dataset has no numbers yet" and "this dataset has
# numbers for the groups that happened to finish". The retrieval log names the shard.
while read -r task dataset subset system k file; do
  [[ -n "$task" ]] || continue
  cell="$task|$dataset|$subset|$system|$file|$k"
  [[ -n "${DISPATCHED[$cell]:-}" ]] && continue
  FAILED["$task/$system"]=1
  log "never dispatched: $task/$system (no merged retrieval file at $file)"
done < <(list_cells)

if (( ${#FAILED[@]} > 0 )); then
  log "cells failed: ${!FAILED[*]}"
fi
log "lane=$LANE done; dispatched=${#DISPATCHED[@]} failed=${#FAILED[@]}"
(( ${#FAILED[@]} == 0 ))
