#!/usr/bin/env bash
#
# run_perf.sh -- comprehensive Linux `perf` profile of ColStore gather/convert.
#
# Drives benchmark/perf_workload.py (a tight single-operation loop, so samples
# land in the kernel under test, not in setup) through the full perf battery and
# saves every artifact under a timestamped results directory:
#
#   perf stat     hardware counters (IPC, cache/branch/TLB miss rates, faults),
#                 plus a best-effort top-down pipeline breakdown.
#   perf record   sampling profile (DWARF call graphs) -> perf.data, kept so a
#   + perf report flamegraph can be made later; flat and call-graph hotspot
#                 listings are written to text.
#   perf annotate per-instruction breakdown of the hottest symbol (which loads /
#                 divides / branches dominate the inner loop).
#   perf mem      where loads are served from (L1/L2/L3/local vs remote DRAM)
#                 and their latency -- the crux of a memory-latency-bound gather.
#   perf c2c      cache-line contention / NUMA false sharing across the OpenMP
#                 threads on a multi-socket box.
#
# Each operation in --ops is profiled separately so the bottleneck of each
# access pattern is visible on its own. The frame() path (typically ~10x the
# dict() path) is in the default set.
#
# Requirements:
#   * perf on PATH (linux-tools; `module load perf` on some clusters).
#   * The extension built with debug info (RelWithDebInfo) so stacks and
#     annotations resolve to source -- build with benchmark/.. or pip install
#     with CMAKE_BUILD_TYPE=RelWithDebInfo first.
#   * perf needs kernel.perf_event_paranoid <= 1 (record/mem/c2c) or CAP_PERFMON;
#     the script checks and warns. mem/c2c additionally need IBS/PEBS support.
#
# Reading the numbers:
#   IPC < 0.5 -> stalled (memory or branch); 1-2 healthy scalar; >3 vectorized.
#   cache-miss / cache-ref > ~50% or high LLC-load-miss -> working set spills to
#   DRAM (expected for scattered gather). High dTLB-load-miss -> huge pages help.
#   perf mem showing many remote-DRAM loads -> NUMA placement is the lever.
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# --- defaults ---------------------------------------------------------------
PYTHON="python3"
WORKLOAD="${SCRIPT_DIR}/perf_workload.py"
OPS=(array-unsorted dict-unsorted frame)
ROWS=10000000
COLS=8
RECORDS=16
INDICES=1000000
SECONDS_PER=3
THREADS=0
FREQ=999
STAT_REPEAT=3
NUMACTL=""
DO_RECORD=1
DO_MEM=1
DO_C2C=1
MEM_OP="array-unsorted"
OUTDIR=""
STORE="/tmp/perf_workload.cstore"

usage() { sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

# --- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)      PYTHON="$2"; shift 2 ;;
    --workload)    WORKLOAD="$2"; shift 2 ;;
    --ops)         read -r -a OPS <<< "$2"; shift 2 ;;
    --rows)        ROWS="$2"; shift 2 ;;
    --cols)        COLS="$2"; shift 2 ;;
    --records)     RECORDS="$2"; shift 2 ;;
    --indices)     INDICES="$2"; shift 2 ;;
    --seconds)     SECONDS_PER="$2"; shift 2 ;;
    --threads)     THREADS="$2"; shift 2 ;;
    --freq)        FREQ="$2"; shift 2 ;;
    --stat-repeat) STAT_REPEAT="$2"; shift 2 ;;
    --numactl)     NUMACTL="$2"; shift 2 ;;
    --mem-op)      MEM_OP="$2"; shift 2 ;;
    --quick)       DO_RECORD=0; DO_MEM=0; DO_C2C=0; shift ;;
    --no-record)   DO_RECORD=0; shift ;;
    --no-mem)      DO_MEM=0; shift ;;
    --no-c2c)      DO_C2C=0; shift ;;
    --outdir)      OUTDIR="$2"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "error: unknown option '$1' (try --help)" >&2; exit 2 ;;
  esac
done

[[ -z "${OUTDIR}" ]] && OUTDIR="perf-results-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${OUTDIR}"

# --- preflight --------------------------------------------------------------
command -v perf >/dev/null 2>&1 || { echo "error: perf not found on PATH" >&2; exit 127; }
[[ -f "${WORKLOAD}" ]] || { echo "error: workload not found: ${WORKLOAD}" >&2; exit 2; }

PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unknown)"
if [[ "${PARANOID}" != "unknown" && "${PARANOID}" -gt 1 ]]; then
  echo ">> warning: kernel.perf_event_paranoid=${PARANOID} (> 1)." >&2
  echo "   record/mem/c2c may be restricted. To relax (needs root):" >&2
  echo "     sudo sysctl -w kernel.perf_event_paranoid=1 kernel.kptr_restrict=0" >&2
fi

# Workload args shared by every stage.
WL_ARGS=(--rows "${ROWS}" --cols "${COLS}" --records "${RECORDS}"
         --indices "${INDICES}" --seconds "${SECONDS_PER}"
         --threads "${THREADS}" --store-path "${STORE}" --keep-store)

# --- environment snapshot ---------------------------------------------------
{
  echo "# run_perf.sh environment $(date -u +%FT%TZ)"
  echo "## perf version"; perf --version 2>&1 || true
  echo "## lscpu"; command -v lscpu >/dev/null && lscpu || echo "(lscpu unavailable)"
  echo "## NUMA topology"; command -v numactl >/dev/null && numactl -H || echo "(numactl unavailable)"
  echo "## perf_event_paranoid = ${PARANOID}"
  echo "## colstore"; "${PYTHON}" -c \
    'import colstore;print("cpp_available",colstore.cpp_available(),"max_threads",colstore.max_threads())' 2>&1 || true
} > "${OUTDIR}/environment.txt"

# Warn (don't fail) if the extension looks stripped: stacks won't symbolise.
EXT_SO="$("${PYTHON}" - <<'PY'
import glob, os
try:
    import colstore
    d = os.path.dirname(colstore.__file__)
    hits = glob.glob(os.path.join(d, "**", "*.so"), recursive=True)
    print(hits[0] if hits else "")
except Exception:
    print("")
PY
)"
if [[ -n "${EXT_SO}" ]] && command -v file >/dev/null 2>&1; then
  file "${EXT_SO}" | grep -q "not stripped" \
    && echo ">> extension has debug symbols: ${EXT_SO}" \
    || echo ">> warning: ${EXT_SO} looks stripped; rebuild RelWithDebInfo for source-level detail" >&2
fi

# Curated, widely-available counters; -d -d adds L1/LLC/dTLB detail. Unsupported
# events degrade to "<not supported>" without failing the run.
STAT_EVENTS="cycles,instructions,branches,branch-misses,cache-references,cache-misses,page-faults,context-switches,cpu-migrations"

# --- prime the store (kept out of every measured region) --------------------
echo ">> priming store (${ROWS} rows x ${COLS} cols x ${RECORDS} records) ..."
"${PYTHON}" "${WORKLOAD}" --op array-unsorted --loops 1 "${WL_ARGS[@]}" >/dev/null 2>>"${OUTDIR}/prime.log" \
  || { echo "error: workload failed to prime (see ${OUTDIR}/prime.log)" >&2; exit 1; }

run_workload=("${PYTHON}" "${WORKLOAD}")
if [[ -n "${NUMACTL}" ]]; then
  # word-split NUMACTL intentionally into separate numactl args
  # shellcheck disable=SC2206
  run_workload=(numactl ${NUMACTL} "${PYTHON}" "${WORKLOAD}")
fi

# --- per-op battery ---------------------------------------------------------
for op in "${OPS[@]}"; do
  echo ">> ===== ${op} ====="

  echo "   perf stat (-r ${STAT_REPEAT}) ..."
  perf stat -d -d -r "${STAT_REPEAT}" -e "${STAT_EVENTS}" \
    -o "${OUTDIR}/stat_${op}.txt" \
    -- "${run_workload[@]}" --op "${op}" "${WL_ARGS[@]}" 2>>"${OUTDIR}/stat_${op}.log" || true

  # Best-effort top-down (Intel: --topdown; AMD/newer: -M TopdownL1). Non-fatal.
  { perf stat --topdown -o "${OUTDIR}/topdown_${op}.txt" \
      -- "${run_workload[@]}" --op "${op}" "${WL_ARGS[@]}" 2>/dev/null \
    || perf stat -M TopdownL1 -o "${OUTDIR}/topdown_${op}.txt" \
      -- "${run_workload[@]}" --op "${op}" "${WL_ARGS[@]}" 2>/dev/null \
    || echo "(top-down metrics unavailable on this perf/CPU)" > "${OUTDIR}/topdown_${op}.txt"; } || true

  if [[ "${DO_RECORD}" -eq 1 ]]; then
    echo "   perf record (-F ${FREQ}, dwarf) ..."
    perf record -F "${FREQ}" -g --call-graph dwarf -o "${OUTDIR}/perf_${op}.data" \
      -- "${run_workload[@]}" --op "${op}" "${WL_ARGS[@]}" >/dev/null 2>>"${OUTDIR}/record_${op}.log" || true

    if [[ -s "${OUTDIR}/perf_${op}.data" ]]; then
      perf report -i "${OUTDIR}/perf_${op}.data" --stdio --no-children --percent-limit 0.5 \
        > "${OUTDIR}/report_flat_${op}.txt" 2>/dev/null || true
      perf report -i "${OUTDIR}/perf_${op}.data" --stdio --children --percent-limit 0.5 \
        > "${OUTDIR}/report_callgraph_${op}.txt" 2>/dev/null || true

      # Annotate the hottest symbol.
      TOP_SYM="$(awk '!/^#/ && NF { sub(/.*\] /, ""); print; exit }' \
        "${OUTDIR}/report_flat_${op}.txt" 2>/dev/null || true)"
      if [[ -n "${TOP_SYM}" ]]; then
        echo "   perf annotate: ${TOP_SYM}"
        perf annotate -i "${OUTDIR}/perf_${op}.data" --stdio --symbol="${TOP_SYM}" \
          > "${OUTDIR}/annotate_${op}.txt" 2>/dev/null || true
      fi
    fi
  fi
done

# --- memory access profile (one op) -----------------------------------------
if [[ "${DO_MEM}" -eq 1 ]]; then
  echo ">> perf mem (${MEM_OP}) ..."
  if perf mem record -o "${OUTDIR}/mem.data" \
       -- "${run_workload[@]}" --op "${MEM_OP}" "${WL_ARGS[@]}" >/dev/null 2>>"${OUTDIR}/mem.log"; then
    perf mem report -i "${OUTDIR}/mem.data" --stdio > "${OUTDIR}/mem_report.txt" 2>/dev/null || true
  else
    echo "(perf mem unavailable -- needs IBS/PEBS and paranoid<=1; see mem.log)" \
      > "${OUTDIR}/mem_report.txt"
  fi
fi

# --- cache-to-cache / false sharing (one op, threaded) ----------------------
if [[ "${DO_C2C}" -eq 1 ]]; then
  echo ">> perf c2c (${MEM_OP}) ..."
  if perf c2c record -o "${OUTDIR}/c2c.data" \
       -- "${run_workload[@]}" --op "${MEM_OP}" "${WL_ARGS[@]}" >/dev/null 2>>"${OUTDIR}/c2c.log"; then
    perf c2c report -i "${OUTDIR}/c2c.data" --stdio > "${OUTDIR}/c2c_report.txt" 2>/dev/null || true
  else
    echo "(perf c2c unavailable -- needs HITM sampling support; see c2c.log)" \
      > "${OUTDIR}/c2c_report.txt"
  fi
fi

# --- clean up the primed store ----------------------------------------------
rm -f "${STORE}"

# --- digest -----------------------------------------------------------------
DIGEST="${OUTDIR}/SUMMARY.txt"
{
  echo "================ perf digest ================"
  echo "results: ${OUTDIR}"
  echo "store:   ${ROWS} rows x ${COLS} cols x ${RECORDS} records, ${INDICES} indices"
  [[ -n "${NUMACTL}" ]] && echo "numactl: ${NUMACTL}"
  echo
  for op in "${OPS[@]}"; do
    echo "---- ${op} ----"
    if [[ -f "${OUTDIR}/stat_${op}.txt" ]]; then
      grep -E "insn per cycle|of all cache refs|of all (L1-dcache|LL-cache|dTLB) " \
        "${OUTDIR}/stat_${op}.txt" | sed 's/^/  /' || true
    fi
    if [[ -f "${OUTDIR}/report_flat_${op}.txt" ]]; then
      echo "  top symbols:"
      awk '!/^#/ && NF { print "    " $0 }' "${OUTDIR}/report_flat_${op}.txt" | head -5 || true
    fi
    echo
  done
  echo "artifacts: stat_*, topdown_*, perf_*.data (-> flamegraph),"
  echo "           report_flat_*, report_callgraph_*, annotate_*,"
  echo "           mem_report.txt, c2c_report.txt, environment.txt"
} | tee "${DIGEST}"

echo
echo ">> done. Full digest: ${DIGEST}"
echo ">> make a flamegraph later from any perf_<op>.data, e.g.:"
echo "     perf script -i ${OUTDIR}/perf_${OPS[0]}.data | stackcollapse-perf.pl | flamegraph.pl > fg.svg"
