#!/usr/bin/env python3
"""Generate the Parquet files that io/parquet's tests read.

    pip install pyarrow
    python tools/parquet/gen.py --out io/parquet/testdata

Every file holds the same table of n rows, whose row i is

    i32   int32    7*i - 100
    i64   int64    i * 10_000_000_000 - 5
    f32   float    i / 3
    f64   double   sqrt(i)
    flag  boolean  i % 3 == 0
    name  string   "s" + str(i % 17)
    opt   float    null when i % 5 == 0, else i / 2

so a test computes the expected contents from the file name alone. The
files differ in how the table is stored:

    plain_v1.parquet      format 1.0, PLAIN, uncompressed, one row group
    dict_v1_snappy        format 1.0, PLAIN_DICTIONARY, Snappy
    dict_v2_gzip          format 2.6, RLE_DICTIONARY, gzip
    dict_v2_zstd          format 2.6, RLE_DICTIONARY, Zstandard
    delta_v2              format 2.6, no dictionary: DELTA_BINARY_PACKED
                          ints, BYTE_STREAM_SPLIT floats,
                          DELTA_LENGTH_BYTE_ARRAY strings
    delta_byte_array_v2   format 2.6, DELTA_BYTE_ARRAY strings
    pages_v2_snappy       format 2.6, small pages, three row groups
    empty                 the schema with no rows

The format version ("v1" or "v2" in the names) decides which encodings
pyarrow may use. Every file holds data pages V1, pyarrow's default;
data_page_version="2.0" would be needed for data pages V2. Every column
is optional, so every page carries definition levels, though only opt
has nulls. The row count n is 1000, except pages_v2_snappy and empty.
"""

import argparse
import math
import os

import pyarrow as pa
import pyarrow.parquet as pq


def table(n: int) -> pa.Table:
    rows = range(n)
    return pa.table(
        {
            "i32": pa.array([7 * i - 100 for i in rows], pa.int32()),
            "i64": pa.array([i * 10_000_000_000 - 5 for i in rows], pa.int64()),
            "f32": pa.array([i / 3 for i in rows], pa.float32()),
            "f64": pa.array([math.sqrt(i) for i in rows], pa.float64()),
            "flag": pa.array([i % 3 == 0 for i in rows], pa.bool_()),
            "name": pa.array(["s" + str(i % 17) for i in rows], pa.string()),
            "opt": pa.array([None if i % 5 == 0 else i / 2 for i in rows], pa.float32()),
        }
    )


FILES = {
    "plain_v1": dict(n=1000, version="1.0", use_dictionary=False, compression="none"),
    "dict_v1_snappy": dict(n=1000, version="1.0", use_dictionary=True, compression="snappy"),
    "dict_v2_gzip": dict(n=1000, version="2.6", use_dictionary=True, compression="gzip"),
    "dict_v2_zstd": dict(n=1000, version="2.6", use_dictionary=True, compression="zstd"),
    "delta_v2": dict(
        n=1000, version="2.6", use_dictionary=False, compression="none",
        column_encoding={
            "i32": "DELTA_BINARY_PACKED", "i64": "DELTA_BINARY_PACKED",
            "f32": "BYTE_STREAM_SPLIT", "f64": "BYTE_STREAM_SPLIT",
            "name": "DELTA_LENGTH_BYTE_ARRAY", "opt": "BYTE_STREAM_SPLIT",
        },
    ),
    "delta_byte_array_v2": dict(
        n=1000, version="2.6", use_dictionary=False, compression="none",
        column_encoding={"name": "DELTA_BYTE_ARRAY"},
    ),
    "pages_v2_snappy": dict(
        n=10000, version="2.6", use_dictionary=True, compression="snappy",
        row_group_size=4000, data_page_size=2048,
    ),
    "empty": dict(n=0, version="2.6", use_dictionary=True, compression="snappy"),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "io", "parquet", "testdata"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, opts in FILES.items():
        opts = dict(opts)
        n = opts.pop("n")
        row_group_size = opts.pop("row_group_size", None)
        path = os.path.join(args.out, name + ".parquet")
        pq.write_table(table(n), path, row_group_size=row_group_size, write_statistics=False, **opts)
        meta = pq.read_metadata(path)
        encodings = sorted({e for g in range(meta.num_row_groups) for c in range(meta.num_columns)
                            for e in meta.row_group(g).column(c).encodings})
        print(f"{name}: {meta.num_rows} rows, {meta.num_row_groups} row groups, {os.path.getsize(path)} bytes, {encodings}")


if __name__ == "__main__":
    main()
