#!/usr/bin/env bash
# Hardware-counter profile of ColStore gather operations.
#
# Requires `linux-tools-generic` (provides perf). Run as root (perf needs
# CAP_PERFMON or kernel.perf_event_paranoid <= 1).
#
# Reads to look for:
#   * IPC (insn per cycle): < 0.5 means stalled (memory or branch);
#     1.0-2.0 is healthy scalar code; > 3.0 means vectorized.
#   * cache-misses / cache-references: < 5% is good, > 50% means the
#     working set doesn't fit in cache.
#   * dTLB-load-misses: high counts here mean huge-pages would help.
#   * page-faults: huge counts during the measured region indicate
#     the store isn't memory-resident; major-faults > 0 means disk.
#
# perf record/report adds hotspot identification (which line of which
# function dominates). Use perf script | ... | flamegraph.pl for visuals.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
WORKLOAD="${1:-${SCRIPT_DIR}/profile_gather.py}"

echo "=== perf stat (CPU counters) ==="
perf stat -d -d \
    -e cycles,instructions,cache-references,cache-misses \
    -e dTLB-loads,dTLB-load-misses \
    -e page-faults,minor-faults,major-faults \
    -e context-switches,cpu-migrations \
    python3 "${WORKLOAD}"

echo
echo "=== perf record (hotspot profile) ==="
perf record -F 999 -g --call-graph dwarf -o /tmp/perf_gather.data \
    python3 "${WORKLOAD}"

echo
echo "=== perf report (top symbols by sample count) ==="
perf report -i /tmp/perf_gather.data --stdio --no-children --max-stack=4 \
    | head -50
