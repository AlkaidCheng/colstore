"""Pin the documented contract of the reader's ``backend`` parameter.

``backend`` selects the kernel for single-record fancy-index reads.
Multi-record stores require the compiled C++ extension and always use it for
fancy reads -- this has been true since multi-record support landed (the
original implementation called the C++ byte-offset gather directly), and the
optimization series preserved it. This test makes that contract explicit so
any future change to it is a conscious one.
"""

from __future__ import annotations

import numpy as np
import pytest

import colstore
from colstore.kernels import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")


def test_multirecord_reads_work_under_numpy_backend(tmp_path):
    rng = np.random.default_rng(8)
    total = 4_000
    full = {"f8": rng.standard_normal(total), "i4": rng.integers(0, 99, total).astype(np.int32)}
    path = tmp_path / "m.cstore"
    with colstore.create(path) as writer:
        for offset in range(0, total, 400):
            writer.write({k: v[offset : offset + 400] for k, v in full.items()})

    dataset = colstore.open(path, backend="numpy")
    assert dataset.backend == "numpy"
    indices = rng.integers(0, total, size=300).astype(np.int64)
    # Single column (unsorted fancy) and the multi-column bin-reuse route
    # both run -- and run through C++ -- regardless of the backend value.
    assert np.array_equal(dataset[indices, "f8"].array(), full["f8"][indices])
    table = dataset[indices, ["f8", "i4"]].dict()
    assert np.array_equal(table["i4"], full["i4"][indices])
    dataset.close()
