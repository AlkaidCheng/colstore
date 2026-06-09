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
// unsorted, bin-reuse, uniform-layout, and boolean-mask reads. All are
// templated on element size only -- a single set of four instantiations
// (sizes 1/2/4/8) covers every fixed-width numeric dtype plus fixed-width
// strings, datetime64, and timedelta64.
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

// Element-indexed gather: ``output[i]`` is the T at
// ``base + indices[i] * sizeof(T)``. ``base`` need not be T-aligned --
// source loads are alignment-safe (packed record bodies make misaligned
// columns legal). The byte-pointer surface keeps the C wrappers uniform;
// the inner loop is typed.
template <typename T>
void gather_indexed_typed(const std::uint8_t* base,
                          const std::int64_t* indices,
                          std::uint8_t* output,
                          std::ptrdiff_t n_indices,
                          int thread_cap = 0,
                          std::ptrdiff_t prefetch_distance = DEFAULT_PREFETCH_DISTANCE);

// Byte-offset gather: ``output[i]`` is the T at ``base + byte_offsets[i]``.
// Offsets need not be T-aligned; source loads are alignment-safe.
template <typename T>
void gather_bytes_typed(const std::uint8_t* base,
                        const std::int64_t* byte_offsets,
                        std::uint8_t* output,
                        std::ptrdiff_t n_indices,
                        int thread_cap = 0,
                        std::ptrdiff_t prefetch_distance = DEFAULT_PREFETCH_DISTANCE);

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
void copy_multirecord_range(const std::uint8_t* base,
                            std::uint8_t* output,
                            std::int64_t start,
                            std::int64_t stop,
                            const std::int64_t* record_starts_rows,
                            const std::int64_t* record_starts_bytes,
                            const std::int64_t* n_rows_per_record,
                            std::int64_t n_records,
                            std::int64_t col_prefix_bytes,
                            std::int64_t itemsize);

// Fused multi-record fancy gather. ``output[i] = column_value(indices[i])``
// for an arbitrary (unsorted) integer index array. For each index the record
// is located by a branchless binary search over ``record_starts_rows`` (the
// R+1 cumulative row boundaries, tiny and cache-resident), the byte address is
// computed in registers, and the element is loaded -- all in one pass,
// OpenMP-parallel across indices.
//
// Templated on element size only (1/2/4/8 bytes); ``sizeof(T)`` is the column
// itemsize. Caller guarantees native byte order and that every index is in
// ``[0, record_starts_rows[n_records])``. ``col_prefix_bytes`` is the summed
// itemsize of the columns preceding this one in a record body.
template <typename T>
void gather_multirecord_typed(const std::uint8_t* base,
                              const std::int64_t* indices,
                              std::uint8_t* output,
                              std::ptrdiff_t n_indices,
                              const std::int64_t* record_starts_rows,
                              const std::int64_t* record_starts_bytes,
                              const std::int64_t* n_rows_per_record,
                              std::int64_t n_records,
                              std::int64_t col_prefix_bytes,
                              int thread_cap = 0,
                              std::ptrdiff_t prefetch_distance = DEFAULT_PREFETCH_DISTANCE);

// Variant of gather_multirecord_typed that additionally records each index's
// record bin (int32) so subsequent columns sharing the same index set can
// skip the binary search entirely. The binning dominates the fused kernel's
// cost and is identical across columns of one read -- computing it once and
// reusing it is the multi-column win. ``bins`` must have length
// ``n_indices``; requires ``n_records <= INT32_MAX`` (the caller guards).
template <typename T>
void gather_multirecord_bins_typed(const std::uint8_t* base,
                                   const std::int64_t* indices,
                                   std::uint8_t* output,
                                   std::int32_t* bins,
                                   std::ptrdiff_t n_indices,
                                   const std::int64_t* record_starts_rows,
                                   const std::int64_t* record_starts_bytes,
                                   const std::int64_t* n_rows_per_record,
                                   std::int64_t n_records,
                                   std::int64_t col_prefix_bytes,
                                   int thread_cap = 0,
                                   std::ptrdiff_t prefetch_distance = DEFAULT_PREFETCH_DISTANCE);

// Companion: gather one column using record bins computed by
// gather_multirecord_bins_typed for the same ``indices``. No search per
// element -- the bin is a sequential int32 read -- and the prefetch
// look-ahead also reads its bin instead of re-searching.
template <typename T>
void gather_multirecord_withbins_typed(const std::uint8_t* base,
                                       const std::int64_t* indices,
                                       std::uint8_t* output,
                                       const std::int32_t* bins,
                                       std::ptrdiff_t n_indices,
                                       const std::int64_t* record_starts_rows,
                                       const std::int64_t* record_starts_bytes,
                                       const std::int64_t* n_rows_per_record,
                                       std::int64_t col_prefix_bytes,
                                       int thread_cap = 0,
                                       std::ptrdiff_t prefetch_distance = DEFAULT_PREFETCH_DISTANCE);

// Sorted multi-record fancy gather: a linear record walk instead of a
// per-element binary search. Requires ``indices`` to be non-decreasing
// (the caller checks; behavior is undefined otherwise). Each OpenMP thread
// binary-searches the record of the first index in its chunk, then advances
// the record cursor monotonically -- O(K + R) total work, no byte_offsets
// array, no per-record host-language loop.
template <typename T>
void gather_multirecord_sorted_typed(const std::uint8_t* base,
                                     const std::int64_t* indices,
                                     std::uint8_t* output,
                                     std::ptrdiff_t n_indices,
                                     const std::int64_t* record_starts_rows,
                                     const std::int64_t* record_starts_bytes,
                                     const std::int64_t* n_rows_per_record,
                                     std::int64_t n_records,
                                     std::int64_t col_prefix_bytes,
                                     int thread_cap = 0,
                                     std::ptrdiff_t prefetch_distance = DEFAULT_PREFETCH_DISTANCE);

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
template <typename T>
void gather_multirecord_strided_typed(const std::uint8_t* base,
                                      std::uint8_t* output,
                                      std::int64_t start,
                                      std::int64_t step,
                                      std::ptrdiff_t n_out,
                                      const std::int64_t* record_starts_rows,
                                      const std::int64_t* record_starts_bytes,
                                      const std::int64_t* n_rows_per_record,
                                      std::int64_t n_records,
                                      std::int64_t col_prefix_bytes,
                                      int thread_cap = 0,
                                      std::ptrdiff_t prefetch_distance = DEFAULT_PREFETCH_DISTANCE);

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
template <typename T>
void gather_multirecord_uniform_typed(const std::uint8_t* base,
                                      const std::int64_t* indices,
                                      std::uint8_t* output,
                                      std::ptrdiff_t n_indices,
                                      std::int64_t rows_per_record,
                                      std::int64_t record_stride_bytes,
                                      std::int64_t first_body_offset,
                                      std::int64_t n_records,
                                      std::int64_t last_record_rows,
                                      std::int64_t col_prefix_bytes,
                                      int thread_cap = 0,
                                      std::ptrdiff_t prefetch_distance = DEFAULT_PREFETCH_DISTANCE);

// Multi-column companions of gather_multirecord_uniform_typed, mirroring
// the bins/withbins pair: the first column computes each index's record
// arithmetically (one division, plus the partial-tail guard) and records
// it in ``bins``; subsequent columns read the bin -- a sequential int32
// load, cheaper than re-dividing -- and form the address with the same
// affine formula, touching no per-record metadata at all. Same layout
// invariants as the single-column kernel; ``n_records <= INT32_MAX``
// (the caller guards, as for the generic bins pair).
template <typename T>
void gather_multirecord_uniform_bins_typed(const std::uint8_t* base,
                                           const std::int64_t* indices,
                                           std::uint8_t* output,
                                           std::int32_t* bins,
                                           std::ptrdiff_t n_indices,
                                           std::int64_t rows_per_record,
                                           std::int64_t record_stride_bytes,
                                           std::int64_t first_body_offset,
                                           std::int64_t n_records,
                                           std::int64_t last_record_rows,
                                           std::int64_t col_prefix_bytes,
                                           int thread_cap = 0,
                                           std::ptrdiff_t prefetch_distance = DEFAULT_PREFETCH_DISTANCE);

template <typename T>
void gather_multirecord_uniform_withbins_typed(const std::uint8_t* base,
                                               const std::int64_t* indices,
                                               std::uint8_t* output,
                                               const std::int32_t* bins,
                                               std::ptrdiff_t n_indices,
                                               std::int64_t rows_per_record,
                                               std::int64_t record_stride_bytes,
                                               std::int64_t first_body_offset,
                                               std::int64_t n_records,
                                               std::int64_t last_record_rows,
                                               std::int64_t col_prefix_bytes,
                                               int thread_cap = 0,
                                               std::ptrdiff_t prefetch_distance = DEFAULT_PREFETCH_DISTANCE);

// Record-base variant of gather_multirecord_withbins_typed for irregular
// files: the caller precomputes, per column,
//   record_base[r] = record_starts_bytes[r]
//                  + col_prefix_bytes * n_rows_per_record[r]
//                  - record_starts_rows[r] * itemsize
// (an O(R) vectorized pass), and the per-element address collapses to
//   off = record_base[bins[i]] + indices[i] * itemsize
// -- one metadata load instead of three, one multiply-add instead of two
// multiplies and three adds. ``bins`` comes from
// gather_multirecord_bins_typed on the same indices; ``record_base`` has
// one entry per record and must be built with this column's prefix and
// itemsize.
template <typename T>
void gather_multirecord_withbins_rbase_typed(const std::uint8_t* base,
                                             const std::int64_t* indices,
                                             std::uint8_t* output,
                                             const std::int32_t* bins,
                                             std::ptrdiff_t n_indices,
                                             const std::int64_t* record_base,
                                             int thread_cap = 0,
                                             std::ptrdiff_t prefetch_distance = DEFAULT_PREFETCH_DISTANCE);

// Boolean-mask-native gather: reads the (uint8 0/1) mask directly -- 1 byte
// per row, linearly; row order is sorted by construction, so the record
// cursor advances monotonically like the sorted walk. Runs of set bits are
// visible in the mask at no extra cost and are served by memcpy (clipped at
// record boundaries); sparse spans are skipped 8 mask bytes at a time. Two
// internal passes: a parallel per-chunk popcount fixes each thread's output
// offset, then each thread gathers its row range. Returns 0 on success, 1 if
// the mask's selected count does not equal ``n_out`` (nothing is written in
// that case) -- the caller sizes ``output`` with np.count_nonzero and the
// kernel verifies rather than trusting. ``mask`` has one byte per row
// (``n_rows`` total, normalized 0/1 as numpy bool guarantees); other
// arguments match the sorted kernel. Native byte order required.
template <typename T>
int gather_multirecord_mask_typed(const std::uint8_t* base,
                                  const std::uint8_t* mask,
                                  std::uint8_t* output,
                                  std::int64_t n_rows,
                                  std::ptrdiff_t n_out,
                                  const std::int64_t* record_starts_rows,
                                  const std::int64_t* record_starts_bytes,
                                  const std::int64_t* n_rows_per_record,
                                  std::int64_t n_records,
                                  std::int64_t col_prefix_bytes,
                                  int thread_cap = 0,
                                  std::ptrdiff_t prefetch_distance = DEFAULT_PREFETCH_DISTANCE);

}  // namespace colstore

// C-callable wrappers used by the Cython binding. Two families:
//   colstore_gather_indexed_<N>: element-indexed (hot path).
//   colstore_gather_bytes_<N>:   byte-offset (multi-record path).
// where <N> is element size in bytes: 1, 2, 4, or 8.
extern "C" {

// Space-separated names of the optimization toggles compiled in (see
// CMakeLists COLSTORE_TOGGLES); empty string when none.
const char* colstore_build_flags(void);

// Diagnostic A/B entries (use_policy selects policy vs legacy); see
// benchmark/check_policy_gather.py.
int colstore_gather_indexed_variant(const std::uint8_t* base,
                                     const std::int64_t* indices,
                                     std::uint8_t* output, std::ptrdiff_t n,
                                     int itemsize, int use_policy,
                                     int thread_cap, std::ptrdiff_t prefetch_distance);
int colstore_gather_bytes_variant(const std::uint8_t* base,
                                   const std::int64_t* byte_offsets,
                                   std::uint8_t* output, std::ptrdiff_t n,
                                   int itemsize, int use_policy,
                                   int thread_cap, std::ptrdiff_t prefetch_distance);

int colstore_gather_indexed(const std::uint8_t* base,
                               const std::int64_t* indices,
                               std::uint8_t* output,
                               std::ptrdiff_t n, int itemsize, int thread_cap,
                               std::ptrdiff_t prefetch_distance);

int colstore_gather_bytes(const std::uint8_t* base,
                             const std::int64_t* byte_offsets,
                             std::uint8_t* output,
                             std::ptrdiff_t n, int itemsize, int thread_cap,
                               std::ptrdiff_t prefetch_distance);

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

// Fused multi-record fancy gather, one per element size (1/2/4/8 bytes).
// See colstore::gather_multirecord_typed for the addressing contract.
int colstore_gather_multirecord(const std::uint8_t* base,
                                   const std::int64_t* indices,
                                   std::uint8_t* output, std::ptrdiff_t n,
                                   const std::int64_t* record_starts_rows,
                                   const std::int64_t* record_starts_bytes,
                                   const std::int64_t* n_rows_per_record,
                                   std::int64_t n_records,
                                   std::int64_t col_prefix_bytes, int itemsize, int thread_cap,
                               std::ptrdiff_t prefetch_distance);

// Sorted walk; see gather_multirecord_sorted_typed.
int colstore_gather_multirecord_sorted(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int itemsize, int thread_cap,
    std::ptrdiff_t prefetch_distance);

// Strided range walk; see gather_multirecord_strided_typed.
int colstore_gather_multirecord_strided(
    const std::uint8_t* base, std::uint8_t* output,
    std::int64_t start, std::int64_t step, std::ptrdiff_t n_out,
    const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int itemsize, int thread_cap,
    std::ptrdiff_t prefetch_distance);

// Uniform-record arithmetic-bin gather; see gather_multirecord_uniform_typed.
int colstore_gather_multirecord_uniform(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::ptrdiff_t n, std::int64_t rows_per_record, std::int64_t record_stride_bytes,
    std::int64_t first_body_offset, std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int itemsize, int thread_cap, std::ptrdiff_t prefetch_distance);

// Uniform bins/withbins pair; see the *_typed declarations.
int colstore_gather_multirecord_uniform_bins(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::int32_t* bins, std::ptrdiff_t n, std::int64_t rows_per_record,
    std::int64_t record_stride_bytes, std::int64_t first_body_offset,
    std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int itemsize, int thread_cap, std::ptrdiff_t prefetch_distance);
int colstore_gather_multirecord_uniform_withbins(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, std::int64_t rows_per_record,
    std::int64_t record_stride_bytes, std::int64_t first_body_offset,
    std::int64_t n_records, std::int64_t last_record_rows,
    std::int64_t col_prefix_bytes, int itemsize, int thread_cap, std::ptrdiff_t prefetch_distance);

// Bin-reuse pair; see gather_multirecord_bins_typed.
int colstore_gather_multirecord_bins(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int itemsize, int thread_cap,
    std::ptrdiff_t prefetch_distance);
int colstore_gather_multirecord_withbins(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t col_prefix_bytes, int itemsize, int thread_cap, std::ptrdiff_t prefetch_distance);

// Record-base withbins variant; see gather_multirecord_withbins_rbase_typed.
int colstore_gather_multirecord_withbins_rbase(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_base,
    int itemsize, int thread_cap, std::ptrdiff_t prefetch_distance);

// Returns OpenMP's maximum thread count, or 1 if OpenMP is not compiled in.
// Exposed for diagnostics.
int colstore_max_threads();

// Boolean-mask-native gather; see gather_multirecord_mask_typed.
int colstore_gather_multirecord_mask(
    const std::uint8_t* base, const std::uint8_t* mask, std::uint8_t* output,
    std::int64_t n_rows, std::ptrdiff_t n_out, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int itemsize, int thread_cap,
    std::ptrdiff_t prefetch_distance);

}  // extern "C"
