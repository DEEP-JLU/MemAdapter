#!/usr/bin/env bash
# Lane launcher for the retrieval-freeze stage of the external benchmarks.
#
# One lane per dataset, so MemTrapBench and PersistBench advance without waiting
# on each other. Within a lane every (task, system) pair is split into shards that
# run as separate processes -- that is the *only* lever on retrieval wall-clock,
# because a single process holds one store client at a time
# (BASELINE_OPT_MEMORY_CACHE_MAX_ENTRIES=1) and therefore builds stores serially.
# Raising --workers would not help here; raising --shards does.
#
# Shards are dispatched through one global pool (--jobs) rather than per group, so
# the cap is on live store handles machine-wide -- the Windows handle growth that
# forced sharding in the first place. Each shard writes its own part file and is
# independently resumable: re-running a finished shard costs 0 model calls and
# exits in well under a second (verified: 0.47s, retrieved=0, skipped=4).
#
# HEAP, NOT HANDLES, IS THE REAL CAP -- measured, not assumed. A shard process
# holds its own SentenceTransformer over the same on-disk bge-m3. safetensors mmaps
# the weights, so pages are shared and each *additional* process costs about
# 1.0 GiB of physical memory rather than the 2.1 GiB its RSS reports (measured:
# 3 processes -> RSS sum 5.73 GiB but available RAM fell only 3.22 GiB; 4 processes
# -> available fell ~3.9 GiB). On this 15 GiB machine roughly 8 GiB is the user's
# own applications and ~5.1 GiB is free at rest, which caps the pool near 5
# processes even though 12 shards would be the handle-safe number.
#
# So the pool gates on available RAM as well as on --jobs: a unit is spawned only
# when both a slot and --min-free-mib of headroom exist. Set --jobs to the width you
# would *like* and let the gate find the width the machine can actually take; the
# run then needs no babysitting and degrades to a slower pool instead of thrashing.
#
# Usage:
#   run_external_retrieval.sh --lane memtrap --shards 12 --jobs 6 --min-free-mib 1536
#   run_external_retrieval.sh --lane both --limit 2 --shards 2 --jobs 2   # pilot
#   run_external_retrieval.sh --lane persist --dry-run
set -Eeuo pipefail

# When Git Bash is started from PowerShell by its absolute ``bash.exe`` path,
# Windows' PATH is inherited but Git's ``/usr/bin`` is not always prepended.
# The launcher uses standard POSIX helpers (dirname, tr, wc, mktemp), so make
# that runtime invariant explicit instead of requiring callers to launch an
# interactive Git-Bash window first.
export PATH="/usr/bin:/bin:${PATH}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
# Invoked as a module, never by path: build_retrieval.py uses relative imports
# (``from . import dataset_adapters``), so running the file directly leaves
# ``__package__`` empty and every shard dies on ImportError before doing any work.
ENTRY="external_benchmarks.build_retrieval"
LOG_ROOT="$ROOT/external_results/logs/retrieval"

LANE="both"
SHARDS=""
JOBS="6"
MIN_FREE_MIB="1536"
LIMIT=""
SYSTEMS="AMEM,Mem0,naiveRAG"
TASKS=""
ATTEMPTS="3"
DRY_RUN="0"
ALLOW_PARTIAL="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lane)          LANE="$2"; shift 2 ;;
    --shards)        SHARDS="$2"; shift 2 ;;
    --jobs)          JOBS="$2"; shift 2 ;;
    --min-free-mib)  MIN_FREE_MIB="$2"; shift 2 ;;
    --limit)         LIMIT="$2"; shift 2 ;;
    --systems)       SYSTEMS="$2"; shift 2 ;;
    --tasks)         TASKS="$2"; shift 2 ;;
    --attempts)      ATTEMPTS="$2"; shift 2 ;;
    --allow-partial) ALLOW_PARTIAL="1"; shift ;;
    --dry-run)       DRY_RUN="1"; shift ;;
    -h|--help)       sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$LANE" =~ ^(memtrap|persist|both)$ ]] || { echo "--lane must be memtrap|persist|both" >&2; exit 2; }

# Default shard counts follow the work, not the row counts: MemTrapBench is 59,707
# messages of store building against PersistBench's 5,310, so it gets the shards.
# Shard count is only the recoverability/balancing granularity -- --jobs and the RAM
# gate set the actual width -- so it can afford to be generous; the per-shard cost
# is one embedder load (~50s) spread over its rows. MemTrapBench rows vary from 30
# to 80 messages, so finer shards also even out the tail.
if [[ -z "$SHARDS" ]]; then
  case "$LANE" in
    memtrap) SHARDS="12" ;;
    persist) SHARDS="4" ;;
    both)    SHARDS="6" ;;
  esac
fi

list_tasks() {
  # ``tr -d '\r'`` is load-bearing: Python's stdout is in text mode, so on Windows
  # every print() emits CRLF and ``read`` keeps the carriage return. The task name
  # then arrives as "memtrap_poison\r" and every unit dies in task_registry.spec()
  # with a KeyError -- before any store work, but after the pool has spawned.
  #
  # ``both`` interleaves the two datasets rather than concatenating them. Units are
  # dispatched in this order and only a handful run at once, so concatenating would
  # start every PersistBench group behind all of MemTrapBench -- the small dataset
  # would wait out the large one instead of fitting inside its window.
  "$PYTHON" - "$ROOT" "$LANE" "$TASKS" <<'PY' | tr -d '\r'
import sys
sys.path.insert(0, sys.argv[1])
from external_benchmarks import task_registry, dataset_adapters
lane, explicit = sys.argv[2], sys.argv[3]
if explicit:
    tasks = [t.strip() for t in explicit.split(",") if t.strip()]
elif lane == "both":
    memtrap = list(dataset_adapters.iter_tasks("memtrapbench"))
    persist = list(dataset_adapters.iter_tasks("persistbench"))
    tasks = [task for pair in zip(memtrap, persist) for task in pair]
    tasks += memtrap[len(persist):]
else:
    dataset = "memtrapbench" if lane == "memtrap" else "persistbench"
    tasks = list(dataset_adapters.iter_tasks(dataset))
for task in tasks:
    print(task)
PY
}

# One work unit per (task, system, shard). Printed as "task system shard".
#
# Shard-major: every task's shard 0 is dispatched before any task's shard 1. With a
# pool of four or five processes a task-major order would run one group's twelve
# shards to completion before starting the next group, so no group would merge -- and
# therefore no group could start generating -- until the lane was a twelfth of the way
# through. Shard-major also keeps the pool spread across subsystems while stores are
# being built, which is where the wall-clock variance is.
list_units() {
  local task
  IFS=',' read -ra syslist <<< "$SYSTEMS"
  for ((shard = 0; shard < SHARDS; shard++)); do
    while read -r task; do
      [[ -n "$task" ]] || continue
      for system in "${syslist[@]}"; do
        printf '%s %s %s\n' "$task" "$system" "$shard"
      done
    done < <(list_tasks)
  done
}

free_mib() {
  # Fails open: if psutil is missing the gate stops gating rather than blocking the
  # run, because a missing measurement tool must never be the reason nothing starts.
  "$PYTHON" -c "import psutil;print(int(psutil.virtual_memory().available/1048576))" 2>/dev/null \
    || echo 999999
}

run_unit() {
  local task="$1" system="$2" shard="$3" attempt="$4"
  local log_dir="$LOG_ROOT/$task/$system"
  local log_file="$log_dir/shard$(printf '%02d' "$shard").log"
  mkdir -p "$log_dir"

  local extra=()
  [[ -n "$LIMIT" ]] && extra+=(--limit "$LIMIT")

  # On a multi-GPU server, map shards round-robin. Each process then sees its
  # assigned card as cuda:0, matching the vendored memory layers.
  local gpu_env=()
  if [[ -n "${RETRIEVAL_GPU_IDS:-}" ]]; then
    local gpu_ids=()
    local gpu_id
    IFS=',' read -r -a gpu_ids <<< "${RETRIEVAL_GPU_IDS}"
    local gpu_hash
    gpu_hash="$(printf '%s' "$task:$system" | cksum | awk '{print $1}')"
    gpu_id="${gpu_ids[$(( (gpu_hash + shard) % ${#gpu_ids[@]} ))]}"
    gpu_env+=("CUDA_VISIBLE_DEVICES=$gpu_id")
    gpu_env+=("MEMORY_EMBEDDER_DEVICE=cuda")
  elif [[ "${RETRIEVAL_GPU_COUNT:-0}" =~ ^[1-9][0-9]*$ ]]; then
    # Include the task and system in the slot.  A shard-only assignment places
    # every shard-0 task on GPU 0 during the first wave, leaving other cards idle.
    local gpu_hash
    gpu_hash="$(printf '%s' "$task:$system" | cksum | awk '{print $1}')"
    gpu_env+=("CUDA_VISIBLE_DEVICES=$(( (gpu_hash + shard) % RETRIEVAL_GPU_COUNT ))")
    gpu_env+=("MEMORY_EMBEDDER_DEVICE=cuda")
  fi

  # Each SentenceTransformer process otherwise inherits all host CPU cores.
  # With many shard processes this creates thousands of competing OpenMP and
  # BLAS threads, which is slower than a bounded pool even on large servers.
  env "${gpu_env[@]}" \
    OMP_NUM_THREADS="${RETRIEVAL_CPU_THREADS:-4}" \
    MKL_NUM_THREADS="${RETRIEVAL_CPU_THREADS:-4}" \
    OPENBLAS_NUM_THREADS="${RETRIEVAL_CPU_THREADS:-4}" \
    NUMEXPR_NUM_THREADS="${RETRIEVAL_CPU_THREADS:-4}" \
    TOKENIZERS_PARALLELISM="false" \
    PYTHONPATH="$ROOT" "$PYTHON" -u -m "$ENTRY" \
    --task "$task" \
    --system "$system" \
    --shard-index "$shard" \
    --shard-count "$SHARDS" \
    "${extra[@]}" \
    >>"$log_file" 2>&1
}

merge_group() {
  local task="$1" system="$2"
  local log_dir="$LOG_ROOT/$task/$system"
  local extra=()
  [[ "$ALLOW_PARTIAL" == "1" ]] && extra+=(--allow-partial)
  mkdir -p "$log_dir"
  PYTHONPATH="$ROOT" "$PYTHON" -u -m "$ENTRY" \
    --task "$task" \
    --system "$system" \
    --shard-count "$SHARDS" \
    --merge \
    "${extra[@]}" \
    >>"$log_dir/merge.log" 2>&1
}

# --- dry run: the call matrix, no work -------------------------------------
if [[ "$DRY_RUN" == "1" ]]; then
  printf 'lane=%s shards=%s jobs=%s limit=%s systems=%s\n' \
    "$LANE" "$SHARDS" "$JOBS" "${LIMIT:-<all>}" "$SYSTEMS"
  "$PYTHON" - "$ROOT" "$LANE" "$TASKS" "$SHARDS" "$LIMIT" "$SYSTEMS" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from external_benchmarks import task_registry, dataset_adapters
from external_benchmarks.build_retrieval import shard_assignment

lane, explicit, shards, limit, systems = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6].split(",")
dataset = None if lane == "both" else ("memtrapbench" if lane == "memtrap" else "persistbench")
tasks = [t.strip() for t in explicit.split(",") if t.strip()] or list(dataset_adapters.iter_tasks(dataset))

# A-MEM runs two LLM calls per stored note and Mem0 roughly two per message;
# naiveRAG stores locally with bge-m3 and makes none.
CALLS_PER_MSG = {"AMEM": 2.0, "Mem0": 2.0, "naiveRAG": 0.0}
lim = int(limit) if limit else None

print(f"\n{'task':32} {'system':10} {'assigned':>8} {'msgs':>10} {'est calls':>11}")
total_rows = total_calls = 0
for task in tasks:
    spec = task_registry.spec(task)
    total = dataset_adapters.sample_count(task)
    assigned = shard_assignment(total, shards, 0)
    if lim:
        assigned = assigned[:lim]
    for system in systems:
        # Per-unit cost is only exact for shard 0; the rows are contiguous blocks
        # so the sample-size spread across shards is small and this is a budget
        # estimate, not a commitment. Merge units are not shown -- they make no
        # model calls.
        try:
            samples = dataset_adapters.load_samples(task, indices=assigned)
            msgs = sum(s.messages_count for s in samples)
        except Exception:
            msgs = 0
        calls = msgs * CALLS_PER_MSG[system]
        total_rows += len(assigned)
        total_calls += calls
        print(f"{task:32} {system:10} {len(assigned):>8} {msgs:>10} {calls:>11.0f}")
print(f"\n{len(tasks)} tasks x {len(systems)} systems x {shards} shards "
      f"= {len(tasks)*len(systems)*shards} processes")
print(f"shard-0 rows {total_rows}, est calls {total_calls:.0f} "
      f"(a full shard set costs about {shards}x this)")
PY
  exit 0
fi

# --- run -------------------------------------------------------------------
LOG_DIR_ALL="$LOG_ROOT/_lane_$LANE"
mkdir -p "$LOG_DIR_ALL"
LANE_LOG="$LOG_DIR_ALL/lane.log"
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LANE_LOG"; }

UNITS_FILE="$(mktemp)"
trap 'rm -f "$UNITS_FILE"' EXIT
list_units > "$UNITS_FILE"
UNIT_COUNT="$(wc -l < "$UNITS_FILE")"

log "lane=$LANE started; units=$UNIT_COUNT shards=$SHARDS jobs=$JOBS min_free=${MIN_FREE_MIB}MiB limit=${LIMIT:-<all>} systems=$SYSTEMS attempts=$ATTEMPTS"

# Pool over every unit at once with a per-unit retry budget. A unit that exhausts
# its attempts is recorded and the merge still runs -- the completeness guard in
# build_retrieval.py --merge is what decides whether the group is usable, so a
# failed shard surfaces as a hard error at merge time rather than a silent gap.
declare -A FAILED=()
declare -A RUNNING=()
THROTTLED="0"

while read -r task system shard; do
  [[ -n "$task" ]] || continue
  task="${task%$'\r'}"; system="${system%$'\r'}"; shard="${shard%$'\r'}"
  while :; do
    live=0
    for pid in "${!RUNNING[@]}"; do
      kill -0 "$pid" 2>/dev/null && live=$((live + 1)) || unset "RUNNING[$pid]"
    done
    if (( live < JOBS )); then
      free="$(free_mib)"
      if (( free >= MIN_FREE_MIB )); then
        if (( THROTTLED == 1 )); then
          log "resuming: available RAM ${free}MiB is back above ${MIN_FREE_MIB}MiB (live=$live)"
          THROTTLED="0"
        fi
        break
      fi
      if (( live == 0 )); then
        # Nothing is running and the floor is still unmet -- almost always because
        # something outside this run owns the machine. Starting one unit keeps the
        # run moving; refusing to start any would stall it forever.
        log "available RAM ${free}MiB < ${MIN_FREE_MIB}MiB floor but no unit is live; starting one anyway"
        break
      fi
      if (( THROTTLED == 0 )); then
        log "throttled: ${live} unit(s) live, available RAM ${free}MiB < ${MIN_FREE_MIB}MiB"
        THROTTLED="1"
      fi
    fi
    sleep 3
  done

  (
    for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
      if run_unit "$task" "$system" "$shard" "$attempt"; then
        exit 0
      fi
      printf '[%s] %s/%s shard=%s attempt=%s failed; retrying\n' \
        "$(date -Is)" "$task" "$system" "$shard" "$attempt" \
        >>"$LOG_ROOT/$task/$system/shard$(printf '%02d' "$shard").log"
      sleep 20
    done
    exit 1
  ) &
  RUNNING[$!]="$task/$system/$shard"
done < "$UNITS_FILE"

for pid in "${!RUNNING[@]}"; do
  wait "$pid" || FAILED["${RUNNING[$pid]}"]=1
done

if (( ${#FAILED[@]} > 0 )); then
  log "units failed after $ATTEMPTS attempts: ${!FAILED[*]}"
fi

# --- merge -----------------------------------------------------------------
# Serial and cheap; the completeness guard rejects any group that is missing rows.
MERGED=0
FAILED_MERGE=0
while read -r task; do
  [[ -n "$task" ]] || continue
  task="${task%$'\r'}"
  IFS=',' read -ra syslist <<< "$SYSTEMS"
  for system in "${syslist[@]}"; do
    if merge_group "$task" "$system"; then
      MERGED=$((MERGED + 1))
      log "merged $task/$system"
    else
      FAILED_MERGE=$((FAILED_MERGE + 1))
      log "MERGE FAILED $task/$system (see $LOG_ROOT/$task/$system/merge.log)"
    fi
  done
done < <(list_tasks)

log "lane=$LANE done; merged=$MERGED merge_failures=$FAILED_MERGE shard_failures=${#FAILED[@]}"
(( FAILED_MERGE == 0 && ${#FAILED[@]} == 0 ))
