API reference
=============

.. currentmodule:: colstore

Every name below is importable directly from the top-level ``colstore`` package.

Opening & reading
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   open
   ColStoreReader
   ColumnView
   TableView
   info
   schema
   ColStoreInfo

Writing
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   store
   create
   recreate
   update
   ColStoreWriter
   compact

Datasets & shards
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ColStoreDataset
   concat
   append
   appender
   Appender

Editing (frames)
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ColStoreFrame
   col

Format interop
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   convert
   saveas
   to_root
   from_parquet
   from_feather
   from_json
   from_npz
   from_hdf
   from_root

Configuration & diagnostics
---------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   calibrate
   ensure_calibrated
   set_max_workers
   get_max_workers
   set_gather_thread_cap
   get_gather_thread_cap
   set_default_backend
   get_default_backend
   set_default_madvise
   get_default_madvise
   max_threads
   cpp_available
   use_passive_openmp_wait

Exceptions
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   FormatError
   QueryError
