// Public header for the colstore gather kernels.
//
// Two single-record entry points share one size-templated machinery:
//
//   gather_indexed: caller passes element indices; kernel computes byte
//   addresses internally as ``base + indices[i] * sizeof(T)``. This is the
//   hot path for any contiguous gather.
//
//   gather_bytes: caller passes pre-computed byte offsets. Used by the
//   multi-record reader where addresses are non-uniform.
//
// The multi-record kernels below them serve range, strided, sorted,
// unsorted, bin-reuse, and uniform-layout reads. All are templated on
// element size only -- a single set of four instantiations (sizes 1/2/4/8)
// covers every fixed-width numeric dtype plus fixed-width strings,
// datetime64, and timedelta64. A single segment-mask kernel (see
// colstore_gather_segment_mask) serves all boolean-mask reads, single-file
// and multi-file alike, over a segment table.
//
// Measurements and the history behind each kernel are recorded in
// docs/optimization_series.md; comments here state the contracts and the
// present design.

#pragma once

#include <cstddef>
#include <cstdint>

// Portable software-prefetch hint. GCC and Clang expose __builtin_prefetch;
// MSVC has no equivalent builtin but provides _mm_prefetch via xmmintrin.h.
// The MSVC arguments (_MM_HINT_NTA = non-temporal, no cache locality)
// match GCC's "rw=0, locality=0" -- the read-once-and-discard hint we
// want for a scattered gather where the source bytes won't be revisited.
#if defined(_MSC_VER)
#include <xmmintrin.h>
#define COLSTORE_PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_NTA)
#else
#define COLSTORE_PREFETCH(addr) __builtin_prefetch((addr), 0, 0)
#endif

// Portable restrict qualifier. GCC and Clang spell it ``__restrict__``;
// MSVC spells it ``__restrict`` (no trailing underscores). C99 has plain
// ``restrict`` but it's not standard C++ -- every compiler has its own
// non-standard equivalent. The qualifier tells the compiler that pointers
// don't alias, enabling load/store vectorization of the inner loop.
#if defined(_MSC_VER)
#define COLSTORE_RESTRICT __restrict
#else
#define COLSTORE_RESTRICT __restrict__
#endif

namespace colstore {

// Default prefetch distance in elements. Eight iterations ahead is roughly
// the sweet spot on current x86 hardware for scattered memory loads.
constexpr std::ptrdiff_t DEFAULT_PREFETCH_DISTANCE = 8;

// Prefetch-distance semantics, shared by all gather kernels: values > 0
// prefetch that many iterations ahead; 0 (or negative) disables software
// prefetching entirely, which can win when the source is cache-resident
// and the prefetch instructions are pure overhead. The extern "C" wrappers
// forward the caller's value unchanged; "use the default" is resolved at
// the Cython layer so this constant has exactly one authoritative home.

// Below this many indices, the OpenMP fork/join cost outweighs the gather
// work, so the kernel runs serially regardless of cap.
constexpr std::ptrdiff_t PARALLEL_THRESHOLD = 1 << 18;  // 262144

// Above the threshold, scale roughly one thread per this many elements (up to
// the cap). Keeps mid-sized gathers from spinning up the full cap.
constexpr std::ptrdiff_t ELEMENTS_PER_THREAD = 1 << 20;  // 1048576

// Resolve the OpenMP thread count for ``n_indices`` under a caller cap
// (<= 0 means OpenMP maximum). Exposed for tests.
std::ptrdiff_t resolve_thread_count(std::ptrdiff_t n_indices, int cap);

}  // namespace colstore

// C-callable wrappers used by the Cython binding. Two families:
//   colstore_gather_indexed_<N>: element-indexed (hot path).
//   colstore_gather_bytes_<N>:   byte-offset (multi-record path).
// where <N> is element size in bytes: 1, 2, 4, or 8.
extern "C" {

// Space-separated names of the optimization toggles compiled in (see
// CMakeLists COLSTORE_TOGGLES); empty string when none.
const char* colstore_build_flags(void);

// Thread count a kernel would use for n_indices elements under a caller
// cap (cap <= 0 means the OpenMP maximum).
long long colstore_resolve_thread_count(long long n_indices, int cap);

// Element-indexed gather: ``output[i]`` is the T at
// ``base + indices[i] * sizeof(T)``. ``base`` need not be T-aligned --
// source loads are alignment-safe (packed record bodies make misaligned
// columns legal). The byte-pointer surface keeps the C wrappers uniform;
// the inner loop is typed.
int colstore_gather_indexed(const std::uint8_t* base,
                               const std::int64_t* indices,
                               std::uint8_t* output,
                               std::ptrdiff_t n, int itemsize, int thread_cap,
                               std::ptrdiff_t prefetch_distance);

// Byte-offset gather: ``output[i]`` is the T at ``base + byte_offsets[i]``.
// Offsets need not be T-aligned; source loads are alignment-safe.
int colstore_gather_bytes(const std::uint8_t* base,
                             const std::int64_t* byte_offsets,
                             std::uint8_t* output,
                             std::ptrdiff_t n, int itemsize, int thread_cap,
                               std::ptrdiff_t prefetch_distance);

// Contiguous multi-record range copy. Copies global rows ``[start, stop)`` of
// one column out of a multi-record file into ``output``, packed contiguously.
// The column's data for a record ``r`` lives at
//   record_starts_bytes[r] + col_prefix_bytes * n_rows_per_record[r]
// and a global row ``idx`` maps to record ``r`` with
//   record_starts_rows[r] <= idx < record_starts_rows[r + 1].
// The range spans one or more whole records; each overlapping record
// contributes one contiguous ``std::memcpy`` of ``count * itemsize`` bytes.
// This is a raw byte copy: the caller guarantees the on-disk dtype is in the
// host's native byte order (a byte copy cannot byteswap). ``record_starts_rows``
// has ``n_records + 1`` entries; the other two index arrays have ``n_records``.
// Contiguous multi-record range copy (size-agnostic; one memcpy per record).
// See colstore::copy_multirecord_range for the addressing contract.
void colstore_copy_multirecord_range(const std::uint8_t* base,
                                     std::uint8_t* output,
                                     std::int64_t start,
                                     std::int64_t stop,
                                     const std::int64_t* record_starts_rows,
                                     const std::int64_t* record_starts_bytes,
                                     const std::int64_t* n_rows_per_record,
                                     std::int64_t n_records,
                                     std::int64_t col_prefix_bytes,
                                     std::int64_t itemsize);

// Fused multi-FILE fancy gather: the fused multi-record gather one level up.
// A dataset of several files is one global row space; each file contributes
// one or more *segments* (a segment is one record of one file), whose global
// rows are ``[segment_starts_rows[s], segment_starts_rows[s + 1])``.
// ``output[i]`` is the element at global row ``indices[i]``, in one pass: the
// segment is found by the same branchless binary search used for records, and
// the byte address is ``segment_base[s] + indices[i] * itemsize``.
// ``segment_base[s]`` is an absolute byte address -- the file's mmap base plus
// the record's body and column offset, with the segment's global start row
// folded out so a *global* index plugs straight in. The address is absolute
// rather than ``base + offset`` because each segment lives in a different mmap;
// reconstructing a cross-mmap pointer from one base by arithmetic would be
// undefined. Output is contiguous in requested order -- no grouping, no
// reorder, no scratch. ``segment_starts_rows`` has ``n_segments + 1`` entries;
// ``segment_base`` has ``n_segments``. Caller guarantees native byte order and
// every index in ``[0, segment_starts_rows[n_segments])``. Dispatched on
// itemsize (1/2/4/8 bytes; any other returns -1).
int colstore_gather_segment(const std::int64_t* indices,
                              std::uint8_t* output, std::ptrdiff_t n,
                              const std::int64_t* segment_starts_rows,
                              const std::int64_t* segment_base,
                              std::int64_t n_segments, int itemsize, int thread_cap,
                              std::ptrdiff_t prefetch_distance);

// Bin-recording multi-file gather: like colstore_gather_segment, and also
// writes ``bins[i]`` = the segment index found for ``indices[i]``. The segment
// is column-independent, so a multi-column read computes it once here and the
// other columns reuse it via colstore_gather_segment_withbins, skipping the
// search -- the same amortization the single-file bins pair gives across
// records. ``segment_base`` is THIS (first) column's per-segment absolute
// bases; ``bins`` has length ``n`` and requires ``n_segments <= INT32_MAX``
// (the caller guards). Other arguments and the dtype contract match
// colstore_gather_segment.
int colstore_gather_segment_bins(const std::int64_t* indices, std::uint8_t* output,
                                   std::int32_t* bins, std::ptrdiff_t n,
                                   const std::int64_t* segment_starts_rows,
                                   const std::int64_t* segment_base, std::int64_t n_segments,
                                   int itemsize, int thread_cap, std::ptrdiff_t prefetch_distance);

// Companion: gather one column reusing segment ids from
// colstore_gather_segment_bins for the same indices. The segment is a
// sequential int32 read instead of a search (the prefetch look-ahead reads its
// bin too), and the address is ``segment_base[bins[i]] + indices[i] * itemsize``
// with THIS column's bases. No segment boundary array is needed.
int colstore_gather_segment_withbins(const std::int64_t* indices, std::uint8_t* output,
                                       const std::int32_t* bins, std::ptrdiff_t n,
                                       const std::int64_t* segment_base, int itemsize,
                                       int thread_cap, std::ptrdiff_t prefetch_distance);

// Sorted multi-file gather: the fast path for a non-decreasing ``indices``,
// replacing the per-index segment search with a monotonic cursor -- O(K +
// n_segments) comparisons with sequential within-segment access. A cursor walk
// has no search to amortize, so multi-column sorted reads call this per column
// rather than the bins pair. The cursor is order-robust: a backward step
// re-locates by binary search, so the output is correct and in-bounds for any
// order (an inaccurate sortedness hint costs only the re-search, never memory
// safety). Arguments and dtype contract otherwise match colstore_gather_segment.
int colstore_gather_segment_sorted(const std::int64_t* indices, std::uint8_t* output,
                                     std::ptrdiff_t n, const std::int64_t* segment_starts_rows,
                                     const std::int64_t* segment_base, std::int64_t n_segments,
                                     int itemsize, int thread_cap,
                                     std::ptrdiff_t prefetch_distance);

// Uniform-grid multi-file gather: colstore_gather_segment specialized for a
// segment table whose every segment holds the same ``rows_per_segment`` rows (the
// global-last may be partial, <= rows_per_segment). The per-index segment is then
// the closed form ``s = idx / rows_per_segment`` (one magic-reciprocal multiply,
// no binary search), then the same ``segment_base[s] + idx * itemsize`` load --
// ``segment_base`` stays an array because the segments live in different mmaps, so
// no ``segment_starts_rows`` is needed. The caller (the dataset detects the grid)
// guarantees every index in ``[0, n_segments * rows_per_segment)`` and native byte
// order. Dispatched on itemsize (1/2/4/8 bytes; any other returns -1).
int colstore_gather_segment_uniform(const std::int64_t* indices, std::uint8_t* output,
                                      std::ptrdiff_t n, std::int64_t rows_per_segment,
                                      const std::int64_t* segment_base, std::int64_t n_segments,
                                      int itemsize, int thread_cap,
                                      std::ptrdiff_t prefetch_distance);

// Bin-recording uniform-grid gather: colstore_gather_segment_uniform that also
// writes ``bins[i]`` = the segment of ``indices[i]``, so a multi-column read
// computes the grid division once here and the trailing columns reuse it via
// colstore_gather_segment_withbins. ``bins`` has length ``n`` and requires
// ``n_segments <= INT32_MAX`` (the caller guards). Other arguments match
// colstore_gather_segment_uniform.
int colstore_gather_segment_uniform_bins(const std::int64_t* indices, std::uint8_t* output,
                                           std::int32_t* bins, std::ptrdiff_t n,
                                           std::int64_t rows_per_segment,
                                           const std::int64_t* segment_base,
                                           std::int64_t n_segments, int itemsize, int thread_cap,
                                           std::ptrdiff_t prefetch_distance);

// Parallel byte copy of runs into one output buffer. Run ``r`` copies
// ``byte_lengths[r]`` bytes from absolute source address ``src_addrs[r]`` to
// ``output + dst_offsets[r]``. The runs' lengths are split across threads in one
// OpenMP region, load-balanced over bytes so a few large runs still parallelize
// -- so a multi-file whole or forward-slice read fills one buffer without a
// per-file host-language copy. Output positions are independent, so the runs may
// leave gaps (a non-viewable file's region, filled separately). Source addresses
// are absolute because the runs are slices of independent mmaps (a base+offset
// form would be undefined across mappings).
void colstore_parallel_copy_runs(std::uint8_t* output, const std::int64_t* src_addrs,
                                 const std::int64_t* dst_offsets,
                                 const std::int64_t* byte_lengths,
                                 std::int64_t n_runs, int thread_cap);

// Interleave columns into a record (structured) array: the SoA -> AoS transpose.
// Row ``i`` of the output holds each column's element ``i`` at its field offset;
// column ``c``'s element ``i`` is ``src_itemsizes[c]`` bytes at
// ``src_addrs[c] + i * src_itemsizes[c]`` (each column contiguous, absolute
// address so the source may be a memmap), written to
// ``output + i * record_itemsize + field_offsets[c]``. Parallelized over row
// ranges, so each thread writes a contiguous record block -- the row-major order
// streams the output once, where the column-major host assignment streams it
// once per column. Caller guarantees native byte order (a raw field copy cannot
// byteswap).
void colstore_interleave_records(std::uint8_t* output, std::int64_t record_itemsize,
                                 std::int64_t n_rows, const std::int64_t* src_addrs,
                                 const std::int64_t* src_itemsizes,
                                 const std::int64_t* field_offsets,
                                 std::int64_t n_cols, int thread_cap);

// Strided multi-record range gather: ``output[i] = column_value(start + i*step)``
// for ``i`` in ``[0, n_out)``. The row stream is arithmetic, so no index array
// exists at all -- the kernel synthesizes each row in a register. Like the
// sorted kernel, each OpenMP thread binary-searches the record of its chunk's
// first row, then advances the record cursor monotonically (forward for
// ``step > 0``, backward for ``step < 0``) -- O(n_out + R) total work.
// ``step`` must be non-zero (the Cython entry validates); the caller
// guarantees every visited row ``start + i*step`` is in
// ``[0, record_starts_rows[n_records])`` (the reader derives ``start``/``stop``
// from ``slice.indices``, which clamps). Caller guarantees native byte order.
// Strided range walk; see colstore_gather_multirecord_strided.
int colstore_gather_multirecord_strided(
    const std::uint8_t* base, std::uint8_t* output,
    std::int64_t start, std::int64_t step, std::ptrdiff_t n_out,
    const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int itemsize, int thread_cap,
    std::ptrdiff_t prefetch_distance);

// Uniform-record fancy gather: the unsorted fused gather specialized for
// files whose records all have the same row count (the final record may be
// partial) and a constant record byte stride. The record bin is computed
// arithmetically -- ``r = idx / rows_per_record`` -- instead of the
// branchless binary search, and the byte address needs no per-record
// metadata loads at all: full records share one affine formula and the
// (possibly partial) final record gets a single guarded base. The caller
// detects the layout (``reader._detect_uniform_record_layout``) and
// guarantees: ``rows_per_record > 0``, every record except possibly the
// last has exactly ``rows_per_record`` rows, ``record_starts_bytes`` is an
// arithmetic sequence with stride ``record_stride_bytes`` starting at
// ``first_body_offset``, ``0 < last_record_rows <= rows_per_record``,
// every index is in range, and the dtype is native byte order.
// Uniform-record arithmetic-bin gather; see colstore_gather_multirecord_uniform.
int colstore_gather_multirecord_uniform(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::ptrdiff_t n, std::int64_t rows_per_record, std::int64_t record_stride_bytes,
    std::int64_t first_body_offset, std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int itemsize, int thread_cap, std::ptrdiff_t prefetch_distance);

// Multi-column companions of colstore_gather_multirecord_uniform, mirroring
// the bins/withbins pair: the first column computes each index's record
// arithmetically (one division, plus the partial-tail guard) and records
// it in ``bins``; subsequent columns read the bin -- a sequential int32
// load, cheaper than re-dividing -- and form the address with the same
// affine formula, touching no per-record metadata at all. Same layout
// invariants as the single-column kernel; ``n_records <= INT32_MAX``
// (the caller guards, as for the generic bins pair).
int colstore_gather_multirecord_uniform_bins(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::int32_t* bins, std::ptrdiff_t n, std::int64_t rows_per_record,
    std::int64_t record_stride_bytes, std::int64_t first_body_offset,
    std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int itemsize, int thread_cap, std::ptrdiff_t prefetch_distance);
// Bin-consuming companion; contract as for colstore_gather_multirecord_uniform_bins
// above.
int colstore_gather_multirecord_uniform_withbins(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, std::int64_t rows_per_record,
    std::int64_t record_stride_bytes, std::int64_t first_body_offset,
    std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int itemsize, int thread_cap, std::ptrdiff_t prefetch_distance);

// Returns OpenMP's maximum thread count, or 1 if OpenMP is not compiled in.
// Exposed for diagnostics.
int colstore_max_threads();

// Boolean-mask-native gather over a segment table -- the one mask kernel for
// both single-file and multi-file reads (a single-file store is just one record
// per segment; a multi-file dataset stitches all files' records into one table).
// ``segment_base[s]`` is the folded absolute base, so this column's value at row
// ``i`` in segment ``s`` is at ``segment_base[s] + i * itemsize``. ``mask`` is
// one uint8 (0/1) per row, ``n_rows`` total; ``n_out`` is the caller's output
// length (``np.count_nonzero(mask)``). ``output[k]`` is the k-th selected element
// in ascending row order -- byte-identical to numpy mask indexing. Pass 1 counts
// each row-chunk's selected rows for exact lock-free output offsets; pass 2 walks
// each chunk with a monotonic segment cursor, word-at-a-time (skip/copy-run/
// branchless-compact), so the cost is O(n_rows + n_segments) with no per-row
// segment search and no index array. ``segment_starts_rows`` has
// ``n_segments + 1`` entries; ``segment_base`` has ``n_segments``. Native byte
// order required. Returns 0 on success, 1 if ``n_out`` disagrees with the mask
// popcount (writing nothing), -1 on unsupported itemsize.
int colstore_gather_segment_mask(const std::uint8_t* mask, std::uint8_t* output,
                                 std::int64_t n_rows, std::ptrdiff_t n_out,
                                 const std::int64_t* segment_starts_rows,
                                 const std::int64_t* segment_base, std::int64_t n_segments,
                                 int itemsize, int thread_cap,
                                 std::ptrdiff_t prefetch_distance);

// Pin OpenMP worker threads to specific CPUs at runtime. In a parallel region
// of ``n`` threads, worker ``t`` is pinned to ``cpus[t]`` via
// ``sched_setaffinity``. This replicates ``OMP_PROC_BIND``/``OMP_PLACES`` --
// which libgomp reads only at OpenMP initialization, too early for a library
// to set once numpy/BLAS has started the runtime -- but applies it at runtime;
// the binding persists for libgomp's reused thread pool. A negative ``cpus[t]``
// leaves worker ``t`` unpinned. Returns the number of workers successfully
// pinned, 0 for ``n <= 0``, or -1 where unsupported (non-Linux, or built
// without OpenMP). Best-effort: a worker whose ``sched_setaffinity`` fails is
// simply not counted.
int colstore_bind_threads_to_cpus(const int* cpus, int n);

}  // extern "C"
