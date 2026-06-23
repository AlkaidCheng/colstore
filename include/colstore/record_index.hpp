// Native record-index walk: build the per-record index the reader needs at
// open without entering the Python interpreter per record.
//
// A .cstore file is a sequence of records, each [32-byte header][body padded
// to 8 bytes]; headers are interleaved with bodies. The Python equivalent is
// colstore.format.read_record_index, which reads each 32-byte header one at a
// time and validates magic / sequential index / CRC32, accumulating the three
// int64 index arrays. This kernel performs the identical walk in C++ to avoid
// per-record interpreter overhead at open.
//
// It reads header-only: each header is read at its computed offset and the body
// is skipped, so only ~32*R header bytes are touched and the file is never
// memory-mapped (a large file is therefore not demand-faulted page by page).
// Output is byte-identical to the Python walk.

#pragma once

#include <cstdint>

extern "C" {

// Walk ``n_records`` record headers starting at ``data_offset``, filling the
// three caller-allocated int64 output arrays:
//   record_starts_rows  [n_records + 1]  cumulative row counts (rows[0] = 0)
//   record_starts_bytes [n_records]      byte offset of each record body
//   n_rows_per_record   [n_records]      per-record row count
// ``itemsize_sum`` is the sum of the per-column itemsizes (a record body is
// ``align_up(n_rows * itemsize_sum, 8)`` bytes). ``read_chunk`` is the size of
// the reused sliding read buffer in bytes (clamped up to one header): larger
// values amortize the syscall count across more records at the cost of reading
// the interleaved bodies as collateral; ``read_chunk == 32`` reads each header
// in isolation (one positional read per record, header bytes only).
//
// Returns 0 on success. On the first corrupt or truncated record it returns a
// negative code and reports the location through the out-parameters so the
// caller can raise a message matching the Python walk:
//   -1 file could not be opened
//   -2 truncated record header  (err_offset, err_record, err_stored = bytes read)
//   -3 bad record magic         (err_offset, err_record)
//   -4 record index mismatch    (err_offset, err_record, err_stored = stored index)
//   -5 header CRC mismatch       (err_offset, err_record, err_crc_stored, err_crc_actual)
//   -6 truncated final body     (err_offset = expected end, err_record, err_stored = file size)
int colstore_read_record_index(const char* path, std::int64_t data_offset,
                               std::int64_t n_records, std::int64_t itemsize_sum,
                               std::int64_t read_chunk, std::int64_t* record_starts_rows,
                               std::int64_t* record_starts_bytes,
                               std::int64_t* n_rows_per_record,
                               std::int64_t* err_offset, std::int64_t* err_record,
                               std::int64_t* err_stored, std::uint32_t* err_crc_stored,
                               std::uint32_t* err_crc_actual);

}  // extern "C"
