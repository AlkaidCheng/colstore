// Public header for the colstore gather kernel.
//
// Two entry points sharing one size-templated machinery:
//
//   gather_indexed: caller passes element indices; kernel computes byte
//   addresses internally as ``base + indices[i] * sizeof(T)``. This is the
//   hot path for any contiguous gather; matches the cost of the original
//   per-dtype kernel because the inner loop is identical after compilation
//   (T is compile-time known via the size-keyed templates).
//
//   gather_bytes: caller passes pre-computed byte offsets. Used by the
//   multi-record reader where addresses are non-uniform.
//
// Both are templated on element size only -- a single set of four
// instantiations (sizes 1/2/4/8) covers every fixed-width numeric dtype plus
// fixed-width strings, datetime64, and timedelta64.

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

// Element-indexed gather. ``base`` is reinterpreted as ``T*`` and indexed
// directly. Caller guarantees natural T alignment of ``base``. The byte-
// pointer surface keeps the C wrappers uniform; the inner loop is typed.
template <typename T>
void gather_indexed_typed(const std::uint8_t* base,
                          const std::int64_t* indices,
                          std::uint8_t* output,
                          std::ptrdiff_t n_indices,
                          int thread_cap = 0,
                          std::ptrdiff_t prefetch_distance = DEFAULT_PREFETCH_DISTANCE);

// Byte-offset gather. ``output[i] = *(T*)(base + byte_offsets[i])``. Caller
// guarantees byte_offsets[i] points at a T-aligned address inside ``base``.
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
// computed in registers, and the element is loaded -- all in one pass. This
// replaces the NumPy pipeline (searchsorted -> record_id -> within_record ->
// byte_offsets -> gather), whose dominant cost is the searchsorted; here the
// binning is fused into the load and parallelized across indices.
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
// skip the binary search entirely. The binning is 87-93% of the fused
// kernel's cost on the deployment hardware, and it is identical across
// columns of one read -- computing it once and reusing it is the
// multi-column win. ``bins`` must have length ``n_indices``; requires
// ``n_records <= INT32_MAX`` (the caller guards).
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
// array, no per-record host-language loop. Replaces the NumPy
// boundary-partition pipeline whose per-record Python loop measures 79-97%
// of the sorted path at R >= 10^4 records.
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

}  // namespace colstore

// C-callable wrappers used by the Cython binding. Two families:
//   colstore_gather_indexed_<N>: element-indexed (hot path).
//   colstore_gather_bytes_<N>:   byte-offset (multi-record path).
// where <N> is element size in bytes: 1, 2, 4, or 8.
extern "C" {

void colstore_gather_indexed_1(const std::uint8_t* base,
                               const std::int64_t* indices,
                               std::uint8_t* output,
                               std::ptrdiff_t n, int thread_cap,
                               std::ptrdiff_t prefetch_distance);
void colstore_gather_indexed_2(const std::uint8_t* base,
                               const std::int64_t* indices,
                               std::uint8_t* output,
                               std::ptrdiff_t n, int thread_cap,
                               std::ptrdiff_t prefetch_distance);
void colstore_gather_indexed_4(const std::uint8_t* base,
                               const std::int64_t* indices,
                               std::uint8_t* output,
                               std::ptrdiff_t n, int thread_cap,
                               std::ptrdiff_t prefetch_distance);
void colstore_gather_indexed_8(const std::uint8_t* base,
                               const std::int64_t* indices,
                               std::uint8_t* output,
                               std::ptrdiff_t n, int thread_cap,
                               std::ptrdiff_t prefetch_distance);

void colstore_gather_bytes_1(const std::uint8_t* base,
                             const std::int64_t* byte_offsets,
                             std::uint8_t* output,
                             std::ptrdiff_t n, int thread_cap,
                               std::ptrdiff_t prefetch_distance);
void colstore_gather_bytes_2(const std::uint8_t* base,
                             const std::int64_t* byte_offsets,
                             std::uint8_t* output,
                             std::ptrdiff_t n, int thread_cap,
                               std::ptrdiff_t prefetch_distance);
void colstore_gather_bytes_4(const std::uint8_t* base,
                             const std::int64_t* byte_offsets,
                             std::uint8_t* output,
                             std::ptrdiff_t n, int thread_cap,
                               std::ptrdiff_t prefetch_distance);
void colstore_gather_bytes_8(const std::uint8_t* base,
                             const std::int64_t* byte_offsets,
                             std::uint8_t* output,
                             std::ptrdiff_t n, int thread_cap,
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
void colstore_gather_multirecord_1(const std::uint8_t* base,
                                   const std::int64_t* indices,
                                   std::uint8_t* output, std::ptrdiff_t n,
                                   const std::int64_t* record_starts_rows,
                                   const std::int64_t* record_starts_bytes,
                                   const std::int64_t* n_rows_per_record,
                                   std::int64_t n_records,
                                   std::int64_t col_prefix_bytes, int thread_cap,
                               std::ptrdiff_t prefetch_distance);
void colstore_gather_multirecord_2(const std::uint8_t* base,
                                   const std::int64_t* indices,
                                   std::uint8_t* output, std::ptrdiff_t n,
                                   const std::int64_t* record_starts_rows,
                                   const std::int64_t* record_starts_bytes,
                                   const std::int64_t* n_rows_per_record,
                                   std::int64_t n_records,
                                   std::int64_t col_prefix_bytes, int thread_cap,
                               std::ptrdiff_t prefetch_distance);
void colstore_gather_multirecord_4(const std::uint8_t* base,
                                   const std::int64_t* indices,
                                   std::uint8_t* output, std::ptrdiff_t n,
                                   const std::int64_t* record_starts_rows,
                                   const std::int64_t* record_starts_bytes,
                                   const std::int64_t* n_rows_per_record,
                                   std::int64_t n_records,
                                   std::int64_t col_prefix_bytes, int thread_cap,
                               std::ptrdiff_t prefetch_distance);
void colstore_gather_multirecord_8(const std::uint8_t* base,
                                   const std::int64_t* indices,
                                   std::uint8_t* output, std::ptrdiff_t n,
                                   const std::int64_t* record_starts_rows,
                                   const std::int64_t* record_starts_bytes,
                                   const std::int64_t* n_rows_per_record,
                                   std::int64_t n_records,
                                   std::int64_t col_prefix_bytes, int thread_cap,
                               std::ptrdiff_t prefetch_distance);

// Sorted walk; see gather_multirecord_sorted_typed.
void colstore_gather_multirecord_sorted_1(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance);
void colstore_gather_multirecord_sorted_2(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance);
void colstore_gather_multirecord_sorted_4(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance);
void colstore_gather_multirecord_sorted_8(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance);

// Bin-reuse pair; see gather_multirecord_bins_typed.
void colstore_gather_multirecord_bins_1(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance);
void colstore_gather_multirecord_withbins_1(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance);
void colstore_gather_multirecord_bins_2(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance);
void colstore_gather_multirecord_withbins_2(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance);
void colstore_gather_multirecord_bins_4(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance);
void colstore_gather_multirecord_withbins_4(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance);
void colstore_gather_multirecord_bins_8(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t n_records, std::int64_t col_prefix_bytes, int thread_cap,
    std::ptrdiff_t prefetch_distance);
void colstore_gather_multirecord_withbins_8(
    const std::uint8_t* base, const std::int64_t* indices, std::uint8_t* output,
    const std::int32_t* bins, std::ptrdiff_t n, const std::int64_t* record_starts_rows,
    const std::int64_t* record_starts_bytes, const std::int64_t* n_rows_per_record,
    std::int64_t col_prefix_bytes, int thread_cap, std::ptrdiff_t prefetch_distance);

// Returns the OpenMP thread cap that gathers will use, or 1 if OpenMP is
// not compiled in. Exposed for diagnostics.
int colstore_max_threads();

}  // extern "C"
