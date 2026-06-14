#!/usr/bin/env bash
#
# run_valgrind.sh -- detailed Memcheck leak check for the colstore extension.
#
# Drives scripts/valgrind_workload.py under Valgrind's Memcheck tool with
# leak detection turned all the way up, the environment tuned so the C++/OpenMP
# extension's allocations are the only thing in view, and known-benign runtime
# noise filtered through scripts/colstore.supp.
#
# Verdict: after the run the log is analysed automatically by
# scripts/analyze_valgrind.py, which categorises every leak record by origin,
# extracts the colstore-attributable ones to a small companion file, writes a
# readable summary, and sets the exit code to 1 iff a leak is attributable to
# colstore's own native frames. On a Python not built --with-valgrind (conda,
# most system Pythons) the interpreter reports thousands of benign "definitely
# lost" blocks from import/startup that the OS reclaims at exit; colstore.supp
# quiets the bulk and the verdict ignores whatever remains because it does not
# pass through colstore code, so the check stays CI-usable on such Pythons.
# The raw (often >100 MB) log is gzip-compressed afterwards by default.
#
# Memcheck needs the *native* extension, not just the Python sources. Build it
# with debug info first (`--build`, or build by hand) so leak stacks resolve to
# colstore's own source lines rather than `???`.
#
# Usage:
#   scripts/run_valgrind.sh [options]
#
# Options:
#   --build               (re)install the package in editable mode with debug
#                         info (RelWithDebInfo) before running
#   --python PATH         Python interpreter to use (default: python3)
#   --rows N              rows per synthetic store        (default 100000)
#   --cols N              columns per store               (default 4)
#   --records N           records per store               (default 8)
#   --iterations N        repeat the workload N times     (default 3)
#   --threads N           gather thread cap; >1 also runs OpenMP under Memcheck
#                         (default 1 -- serial kernels give the cleanest report)
#   --track-origins       add --track-origins=yes (origins of uninitialised
#                         values; slower, useful when chasing more than leaks)
#   --workload PATH       workload script (default: scripts/valgrind_workload.py)
#   --supp PATH           extra suppression file (repeatable)
#   --gen-suppressions    add --gen-suppressions=all (emit ready-to-vet stanzas;
#                         off by default as it greatly enlarges the log)
#   --top N               show the N most frequent leaking stacks (default 10)
#   --keep-log            leave the raw log uncompressed
#   --delete-log          delete the raw log after analysis (keep summary only)
#   --out PATH            valgrind log file
#                         (default: valgrind-colstore-<timestamp>.log)
#   -h, --help            show this help and exit
#
set -euo pipefail

# --- locate repo root from this script's location ---------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- defaults ---------------------------------------------------------------
PYTHON="python3"
ROWS=100000
COLS=4
RECORDS=8
ITERATIONS=3
THREADS=1
DO_BUILD=0
TRACK_ORIGINS=0
WORKLOAD="${SCRIPT_DIR}/valgrind_workload.py"
ANALYZER="${SCRIPT_DIR}/analyze_valgrind.py"
DEFAULT_SUPP="${SCRIPT_DIR}/colstore.supp"
EXTRA_SUPPS=()
GEN_SUPP=0
TOP=10
LOG_DISPOSITION="compress"   # compress | keep | delete
OUT=""

usage() { sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

# --- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)         DO_BUILD=1; shift ;;
    --python)        PYTHON="$2"; shift 2 ;;
    --rows)          ROWS="$2"; shift 2 ;;
    --cols)          COLS="$2"; shift 2 ;;
    --records)       RECORDS="$2"; shift 2 ;;
    --iterations)    ITERATIONS="$2"; shift 2 ;;
    --threads)       THREADS="$2"; shift 2 ;;
    --track-origins) TRACK_ORIGINS=1; shift ;;
    --workload)      WORKLOAD="$2"; shift 2 ;;
    --supp)          EXTRA_SUPPS+=("$2"); shift 2 ;;
    --gen-suppressions) GEN_SUPP=1; shift ;;
    --top)           TOP="$2"; shift 2 ;;
    --keep-log)      LOG_DISPOSITION="keep"; shift ;;
    --delete-log)    LOG_DISPOSITION="delete"; shift ;;
    --out)           OUT="$2"; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "error: unknown option '$1' (try --help)" >&2; exit 2 ;;
  esac
done

[[ -z "${OUT}" ]] && OUT="${REPO_ROOT}/valgrind-colstore-$(date +%Y%m%d-%H%M%S).log"

# --- preflight --------------------------------------------------------------
if ! command -v valgrind >/dev/null 2>&1; then
  cat >&2 <<'MSG'
error: valgrind not found on PATH.
  Debian/Ubuntu : sudo apt-get install valgrind
  RHEL/Fedora   : sudo dnf install valgrind
  conda         : conda install -c conda-forge valgrind
  Perlmutter    : module load valgrind   (or build from source in $HOME)
MSG
  exit 127
fi
[[ -f "${WORKLOAD}" ]] || { echo "error: workload not found: ${WORKLOAD}" >&2; exit 2; }
[[ -f "${ANALYZER}" ]] || { echo "error: analyzer not found: ${ANALYZER}" >&2; exit 2; }

# --- optional build with debug info -----------------------------------------
if [[ "${DO_BUILD}" -eq 1 ]]; then
  echo ">> building colstore with debug info (RelWithDebInfo) ..."
  ( cd "${REPO_ROOT}" \
    && CMAKE_BUILD_TYPE=RelWithDebInfo \
       SKBUILD_CMAKE_BUILD_TYPE=RelWithDebInfo \
       "${PYTHON}" -m pip install -e . --no-build-isolation -v )
fi

# Warn (don't fail) if the built extension carries no debug symbols: leak
# stacks would point into the .so without source lines.
EXT_SO="$("${PYTHON}" - <<'PY'
import glob, os
try:
    import colstore
    d = os.path.dirname(colstore.__file__)
    hits = glob.glob(os.path.join(d, "_gather*.so")) + glob.glob(os.path.join(d, "**", "*.so"), recursive=True)
    print(hits[0] if hits else "")
except Exception:
    print("")
PY
)"
if [[ -n "${EXT_SO}" ]] && command -v file >/dev/null 2>&1; then
  if file "${EXT_SO}" | grep -q "not stripped"; then
    echo ">> extension has debug symbols: ${EXT_SO}"
  else
    echo ">> warning: ${EXT_SO} looks stripped; rebuild with --build for source-level leak stacks" >&2
  fi
fi

# --- environment: make the extension the only thing Memcheck sees -----------
# PYTHONMALLOC=malloc  -> disable pymalloc so Memcheck tracks real malloc/free
# OMP_*                -> bound, sleeping OpenMP threads (less pool noise)
export PYTHONMALLOC=malloc
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${THREADS}"
export OMP_WAIT_POLICY=passive
export OMP_PROC_BIND=false
export GOMP_SPINCOUNT=0

# --- assemble suppression args (project file + any system CPython file) ------
SUPP_ARGS=()
[[ -f "${DEFAULT_SUPP}" ]] && SUPP_ARGS+=("--suppressions=${DEFAULT_SUPP}")
for cand in \
    "$("${PYTHON}" -c 'import sys,os;print(os.path.join(sys.base_prefix,"share","doc","python%d.%d"%sys.version_info[:2],"valgrind-python.supp"))' 2>/dev/null)" \
    /usr/lib/valgrind/python3.supp \
    /usr/share/doc/python3/valgrind-python.supp; do
  [[ -n "${cand}" && -f "${cand}" ]] && SUPP_ARGS+=("--suppressions=${cand}")
done
for s in "${EXTRA_SUPPS[@]:-}"; do
  [[ -n "${s}" ]] && SUPP_ARGS+=("--suppressions=${s}")
done

# --- assemble Valgrind options ----------------------------------------------
VG_OPTS=(
  --tool=memcheck
  --leak-check=full
  --show-leak-kinds=definite,indirect,possible   # hide 'reachable' (framework noise)
  --errors-for-leak-kinds=definite,indirect       # only real leaks set the exit code
  --error-exitcode=1
  --num-callers=40                                # deep Cython/OpenMP stacks
  --keep-debuginfo=yes                            # symbolise the .so after it's unloaded
  --trace-children=no
  --child-silent-after-fork=yes
  --log-file="${OUT}"
)
[[ "${GEN_SUPP}" -eq 1 ]] && VG_OPTS+=(--gen-suppressions=all)  # large; emit on request
[[ "${TRACK_ORIGINS}" -eq 1 ]] && VG_OPTS+=(--track-origins=yes)
[[ "${THREADS}" -gt 1 ]] && VG_OPTS+=(--fair-sched=yes)

# --- run --------------------------------------------------------------------
echo ">> running Memcheck (this is ~20-50x slower than native; please wait)"
echo ">> log: ${OUT}"
# Valgrind exits non-zero when it finds leaks; the real verdict comes from the
# analyser below (colstore-attributed), so don't let the raw exit abort us here.
set +e
valgrind "${VG_OPTS[@]}" "${SUPP_ARGS[@]}" \
  "${PYTHON}" "${WORKLOAD}" \
    --rows "${ROWS}" --cols "${COLS}" --records "${RECORDS}" \
    --iterations "${ITERATIONS}" --threads "${THREADS}"
set -e

# --- analyse + verdict ------------------------------------------------------
# The raw log is too large to read by hand; analyze_valgrind.py categorises every
# leak record, writes a readable summary and a small file of any
# colstore-attributable records, and returns 1 iff colstore itself leaks.
set +e
"${PYTHON}" "${ANALYZER}" "${OUT}" --top "${TOP}"
ANALYZE_STATUS=$?
set -e

# --- dispose of the raw log -------------------------------------------------
# Everything actionable now lives in the summary and colstore-leaks files; keep
# the raw log only as compressed evidence (default), or per the user's choice.
case "${LOG_DISPOSITION}" in
  compress)
    if command -v gzip >/dev/null 2>&1; then
      gzip -f "${OUT}"
      echo ">> raw log compressed: ${OUT}.gz"
    else
      echo ">> raw log: ${OUT} (gzip unavailable; left uncompressed)"
    fi
    ;;
  delete)
    rm -f "${OUT}"
    echo ">> raw log deleted (summary and colstore-leaks files retained)"
    ;;
  keep)
    echo ">> raw log: ${OUT}"
    ;;
esac

exit "${ANALYZE_STATUS}"
