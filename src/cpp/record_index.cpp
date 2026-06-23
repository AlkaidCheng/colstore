// See include/colstore/record_index.hpp for the contract. This is a faithful
// C++ port of colstore.format.read_record_index's per-record header walk:
// identical validation order (magic, sequential index, CRC32), identical
// body-offset arithmetic, and identical truncation checks, producing
// byte-identical int64 index arrays.
//
// Headers are read through a reused fixed-size buffer that slides forward over
// the file (refilled with one positional read whenever the next header is not
// already buffered). This keeps the syscall count proportional to file size /
// buffer size rather than to the record count, and the small reused buffer is
// faulted in once -- the file itself is never memory-mapped, so a large file is
// not demand-faulted page by page.

#include "colstore/record_index.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

// Standard CRC-32 (zlib / IEEE 802.3): reflected input/output, polynomial
// 0xEDB88320, initial and final XOR 0xFFFFFFFF. This is exactly Python's
// zlib.crc32, so the per-record check accepts/rejects the same bytes; the
// byte-identity tests pin the agreement. Built once at load (single-threaded,
// before any nogil call) and read-only thereafter.
struct Crc32Table {
  std::uint32_t entry[256];
  Crc32Table() {
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k) {
        c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      }
      entry[i] = c;
    }
  }
};

const Crc32Table kCrcTable;

inline std::uint32_t crc32_of(const std::uint8_t* data, int length) {
  std::uint32_t crc = 0xFFFFFFFFu;
  for (int i = 0; i < length; ++i) {
    crc = kCrcTable.entry[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

// Little-endian field reads matching the '<4sqqqI' header layout, independent
// of host byte order (the file format is little-endian).
inline std::int64_t read_le_i64(const std::uint8_t* p) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
  }
  return static_cast<std::int64_t>(v);
}

inline std::uint32_t read_le_u32(const std::uint8_t* p) {
  std::uint32_t v = 0;
  for (int i = 0; i < 4; ++i) {
    v |= static_cast<std::uint32_t>(p[i]) << (8 * i);
  }
  return v;
}

constexpr int kHeaderSize = 32;
constexpr int kCrcCoverage = 28;  // CRC covers all 32 header bytes but the CRC field
constexpr std::int64_t kBodyAlignment = 8;
const std::uint8_t kMagic[4] = {'R', 'E', 'C', 0x01};

inline std::int64_t align_up(std::int64_t value, std::int64_t alignment) {
  return (value + (alignment - 1)) / alignment * alignment;
}

// Thin portable file handle: open read-only, positional read (up to count
// bytes, fewer only at EOF), file size, close.
#ifdef _WIN32
inline int open_ro(const char* path) {
  int fd = -1;
  _sopen_s(&fd, path, _O_RDONLY | _O_BINARY, _SH_DENYNO, 0);
  return fd;
}
inline std::int64_t read_at(int fd, std::uint8_t* dst, std::int64_t count, std::int64_t offset) {
  if (_lseeki64(fd, offset, SEEK_SET) < 0) {
    return -1;
  }
  std::int64_t total = 0;
  while (total < count) {
    int got = _read(fd, dst + total, static_cast<unsigned int>(count - total));
    if (got < 0) {
      return -1;
    }
    if (got == 0) {
      break;
    }
    total += got;
  }
  return total;
}
inline std::int64_t file_size_of(int fd) { return _filelengthi64(fd); }
inline void close_fd(int fd) { _close(fd); }
#else
inline int open_ro(const char* path) { return open(path, O_RDONLY); }
inline std::int64_t read_at(int fd, std::uint8_t* dst, std::int64_t count, std::int64_t offset) {
  std::int64_t total = 0;
  while (total < count) {
    ssize_t got = pread(fd, dst + total, static_cast<size_t>(count - total),
                        static_cast<off_t>(offset + total));
    if (got < 0) {
      return -1;
    }
    if (got == 0) {
      break;
    }
    total += got;
  }
  return total;
}
inline std::int64_t file_size_of(int fd) {
  struct stat st;
  if (fstat(fd, &st) != 0) {
    return -1;
  }
  return static_cast<std::int64_t>(st.st_size);
}
inline void close_fd(int fd) { close(fd); }
#endif

struct FdGuard {
  int fd;
  ~FdGuard() {
    if (fd >= 0) {
      close_fd(fd);
    }
  }
};

}  // namespace

extern "C" int colstore_read_record_index(
    const char* path, std::int64_t data_offset, std::int64_t n_records,
    std::int64_t itemsize_sum, std::int64_t read_chunk, std::int64_t* record_starts_rows,
    std::int64_t* record_starts_bytes, std::int64_t* n_rows_per_record,
    std::int64_t* err_offset, std::int64_t* err_record, std::int64_t* err_stored,
    std::uint32_t* err_crc_stored, std::uint32_t* err_crc_actual) {
  int fd = open_ro(path);
  if (fd < 0) {
    return -1;
  }
  FdGuard guard{fd};

  if (read_chunk < kHeaderSize) {
    read_chunk = kHeaderSize;  // the buffer must hold at least one full header
  }
  std::vector<std::uint8_t> buffer(static_cast<std::size_t>(read_chunk));
  std::int64_t buf_start = 0;  // buffer covers [buf_start, buf_start + buf_len)
  std::int64_t buf_len = 0;

  record_starts_rows[0] = 0;
  std::int64_t cumulative_rows = 0;
  std::int64_t next_offset = data_offset;

  for (std::int64_t record = 0; record < n_records; ++record) {
    // Refill the sliding buffer whenever the 32-byte header at next_offset is
    // not already fully buffered (the common case is a hit, no syscall).
    if (next_offset < buf_start || next_offset + kHeaderSize > buf_start + buf_len) {
      std::int64_t got = read_at(fd, buffer.data(), read_chunk, next_offset);
      if (got < 0) {
        return -1;
      }
      buf_start = next_offset;
      buf_len = got;
    }
    if (next_offset + kHeaderSize > buf_start + buf_len) {
      *err_offset = next_offset;
      *err_record = record;
      *err_stored = buf_start + buf_len - next_offset;  // bytes actually available
      return -2;
    }

    const std::uint8_t* header = buffer.data() + (next_offset - buf_start);
    if (std::memcmp(header, kMagic, 4) != 0) {
      *err_offset = next_offset;
      *err_record = record;
      return -3;
    }
    const std::int64_t stored_index = read_le_i64(header + 4);
    if (stored_index != record) {
      *err_offset = next_offset;
      *err_record = record;
      *err_stored = stored_index;
      return -4;
    }
    const std::uint32_t stored_crc = read_le_u32(header + 28);
    const std::uint32_t actual_crc = crc32_of(header, kCrcCoverage);
    if (actual_crc != stored_crc) {
      *err_offset = next_offset;
      *err_record = record;
      *err_crc_stored = stored_crc;
      *err_crc_actual = actual_crc;
      return -5;
    }

    const std::int64_t n_rows = read_le_i64(header + 12);
    const std::int64_t body_offset = next_offset + kHeaderSize;
    record_starts_bytes[record] = body_offset;
    n_rows_per_record[record] = n_rows;
    cumulative_rows += n_rows;
    record_starts_rows[record + 1] = cumulative_rows;
    next_offset = body_offset + align_up(n_rows * itemsize_sum, kBodyAlignment);
  }

  // The file must extend to the end of the last record's padded body. An
  // inter-record truncation is already caught by the header read above; a
  // truncation inside the final record's body is only visible here.
  const std::int64_t file_size = file_size_of(fd);
  if (file_size >= 0 && file_size < next_offset) {
    *err_offset = next_offset;
    *err_record = n_records - 1;
    *err_stored = file_size;
    return -6;
  }
  return 0;
}
