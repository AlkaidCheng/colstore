"""Parse human-readable byte-size strings using IEC 80000-13 units.

Decimal SI prefixes (``kB``, ``MB``, ``GB``, ``TB``, ``PB``) are powers of
1000; IEC binary prefixes (``KiB``, ``MiB``, ``GiB``, ``TiB``, ``PiB``) are
powers of 1024. The two are distinct, per IEC 80000-13 (adopted by IEEE 1541,
ISO, and NIST): ``1 MB`` is 1,000,000 bytes and ``1 MiB`` is 1,048,576 bytes.

Parsing is case-insensitive and the presence of an ``i`` selects the binary
interpretation, so ``"MiB"``, ``"mib"``, and ``"MIB"`` all mean 1024**2. A bare
prefix without a trailing ``B`` (``"4M"``) is read as decimal, and a bare
number (``"4096"``) is bytes.
"""

from __future__ import annotations

import re

_DECIMAL_STEP = 1000
_BINARY_STEP = 1024

# Power of each prefix: kilo=1, mega=2, giga=3, tera=4, peta=5.
_PREFIX_POWERS = {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}


def _build_unit_factors() -> dict[str, int]:
    """Map every accepted (upper-cased) unit spelling to its byte multiplier."""
    factors = {"": 1, "B": 1}
    for prefix, power in _PREFIX_POWERS.items():
        decimal = _DECIMAL_STEP**power
        binary = _BINARY_STEP**power
        factors[prefix] = decimal  # "M"  -> decimal
        factors[prefix + "B"] = decimal  # "MB" -> decimal
        factors[prefix + "I"] = binary  # "MI" -> binary
        factors[prefix + "IB"] = binary  # "MIB" -> binary
    return factors


_UNIT_FACTORS = _build_unit_factors()
_BYTE_SIZE_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*$")


def parse_byte_size(size: str) -> int:
    """Parse a size string like ``"100 MB"`` or ``"1.5 GiB"`` into bytes.

    Units follow IEC 80000-13: decimal SI prefixes (``kB``, ``MB``, ``GB``,
    ``TB``, ``PB``) are powers of 1000 and IEC binary prefixes (``KiB``,
    ``MiB``, ``GiB``, ``TiB``, ``PiB``) are powers of 1024. So ``"1 MB"`` is
    1,000,000 and ``"1 MiB"`` is 1,048,576.

    Parameters
    ----------
    size : str
        A number with an optional unit. Examples: ``"100 MB"``, ``"10KiB"``,
        ``"2.5 GiB"``, ``"4096"``. Case-insensitive; surrounding whitespace is
        ignored. A bare prefix (``"4M"``) is decimal; a bare number is bytes.

    Returns
    -------
    int
        The size in bytes, always at least 1.

    Raises
    ------
    ValueError
        If the string cannot be parsed, the unit is unrecognized, or the
        result rounds to less than one byte.
    """
    match = _BYTE_SIZE_PATTERN.match(size)
    if not match:
        raise ValueError(
            f"Cannot parse byte size {size!r}; expected a number with optional unit "
            f"(e.g., '100 MB', '1.5 GiB', '4 KiB')."
        )
    value = float(match.group(1))
    unit = match.group(2).upper()
    if unit not in _UNIT_FACTORS:
        supported = sorted(u for u in _UNIT_FACTORS if u)
        raise ValueError(
            f"Unknown unit {match.group(2)!r} in byte size {size!r}; "
            f"expected one of {supported}."
        )
    result = int(value * _UNIT_FACTORS[unit])
    if result < 1:
        raise ValueError(f"Byte size must be at least 1 byte; {size!r} resolves to {result}.")
    return result


def resolve_batch_rows(
    batch_size: int | str | None,
    *,
    bytes_per_row: int | None = None,
) -> int | None:
    """Resolve a polymorphic ``batch_size`` to a number of rows per batch.

    Parameters
    ----------
    batch_size : int, str, or None
        ``None`` streams everything in a single batch. An ``int`` is a row
        count. A ``str`` is a memory budget (see :func:`parse_byte_size`)
        converted to rows using ``bytes_per_row``.
    bytes_per_row : int or None, optional
        Bytes occupied by one row, required only when ``batch_size`` is a
        ``str``. Must be positive.

    Returns
    -------
    int or None
        Rows per batch, or ``None`` for a single-batch pass.

    Raises
    ------
    ValueError
        If an ``int`` ``batch_size`` is not positive, or a ``str`` budget is
        given without a positive ``bytes_per_row``.
    TypeError
        If ``batch_size`` is not int, str, or None.
    """
    if batch_size is None:
        return None
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, str)):
        raise TypeError(f"batch_size must be int, str, or None; got {type(batch_size).__name__}.")
    if isinstance(batch_size, int):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}.")
        return batch_size
    budget = parse_byte_size(batch_size.strip())
    if bytes_per_row is None or bytes_per_row <= 0:
        raise ValueError("A string batch_size needs a positive bytes_per_row to size the budget.")
    return max(1, budget // bytes_per_row)
