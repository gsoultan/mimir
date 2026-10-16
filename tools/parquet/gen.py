#!/usr/bin/env python3
"""Generate the Parquet files that io/parquet's tests read.

    pip install pyarrow
    python tools/parquet/gen.py --out io/parquet/testdata --shards data/datasets/testdata

Every file under --out holds the same table of n rows, whose row i is

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
pyarrow may use. The v1 files hold data pages V1, pyarrow's default, and
the v2 files data pages V2 (data_page_version="2.0"), whose levels stay
uncompressed before the compressed values. Every column is optional, so
every page carries definition levels, though only opt has nulls. The
row count n is 1000, except pages_v2_snappy and empty.

The files under --shards are what datasets.ParquetShards's tests read:
shards of a regression table, whose row i (counted over all its files) is

    x1  float    i / 4
    x2  int32    i % 7 - 3
    x3  boolean  i % 2 == 0
    x4  int64    i * 1_000_000_007
    y   double   i / 2 + 0.5

and of a classification table, whose row i is

    x1     float   i / 4
    x2     int32   i % 7 - 3
    label  string  "abc"[i % 3]

    reg_0        rows 0 to 99 of the regression table, two row groups
    reg_1        rows 100 to 129, one row group
    cls_0        rows 0 to 99 of the classification table, two row groups,
                 dictionary encoded, data pages V2
    cls_1        rows 100 to 129, one row group, plain
    nulls        rows 0 to 9 of the regression table with x2 null at row 3
    mismatch     rows 0 to 9 of the regression table with x2 as int64
    single       rows 0 to 9 of y alone
    required     rows 0 to 9 of the regression table with x1 required
                 (nullable=False), the other columns optional
    x1_null      rows 10 to 19, every column optional, x1 null at row 14
    nested       rows 0 to 9 of the regression table with a column vals of
                 list<int32>, [i, i + 1] in row i
    logical      two rows of logical types stored as integers, and y 1, 2:
                 d64 decimal(10, 2) 12.34, 0.01 as INT64
                 d32 decimal(5, 3) -1.234, 0.5 as INT32
                 day date32 1970-01-11, 2000-01-01 (days 10, 10957)
    no_rows      x1 and y with no row groups at all, as a writer closed
                 without writing leaves a file
"""

import argparse
import datetime
import math
import os
from decimal import Decimal

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
    "dict_v2_gzip": dict(n=1000, version="2.6", use_dictionary=True, compression="gzip", data_page_version="2.0"),
    "dict_v2_zstd": dict(n=1000, version="2.6", use_dictionary=True, compression="zstd", data_page_version="2.0"),
    "delta_v2": dict(
        n=1000, version="2.6", use_dictionary=False, compression="none", data_page_version="2.0",
        column_encoding={
            "i32": "DELTA_BINARY_PACKED", "i64": "DELTA_BINARY_PACKED",
            "f32": "BYTE_STREAM_SPLIT", "f64": "BYTE_STREAM_SPLIT",
            "name": "DELTA_LENGTH_BYTE_ARRAY", "opt": "BYTE_STREAM_SPLIT",
        },
    ),
    "delta_byte_array_v2": dict(
        n=1000, version="2.6", use_dictionary=False, compression="none", data_page_version="2.0",
        column_encoding={"name": "DELTA_BYTE_ARRAY"},
    ),
    "pages_v2_snappy": dict(
        n=10000, version="2.6", use_dictionary=True, compression="snappy", data_page_version="2.0",
        row_group_size=4000, data_page_size=2048,
    ),
    "empty": dict(n=0, version="2.6", use_dictionary=True, compression="snappy"),
}


def regression(start: int, n: int, x2: pa.DataType = pa.int32(), x2_null: int = -1, x1_null: int = -1) -> pa.Table:
    rows = range(start, start + n)
    return pa.table(
        {
            "x1": pa.array([None if i == x1_null else i / 4 for i in rows], pa.float32()),
            "x2": pa.array([None if i == x2_null else i % 7 - 3 for i in rows], x2),
            "x3": pa.array([i % 2 == 0 for i in rows], pa.bool_()),
            "x4": pa.array([i * 1_000_000_007 for i in rows], pa.int64()),
            "y": pa.array([i / 2 + 0.5 for i in rows], pa.float64()),
        }
    )


def classification(start: int, n: int) -> pa.Table:
    rows = range(start, start + n)
    return pa.table(
        {
            "x1": pa.array([i / 4 for i in rows], pa.float32()),
            "x2": pa.array([i % 7 - 3 for i in rows], pa.int32()),
            "label": pa.array(["abc"[i % 3] for i in rows], pa.string()),
        }
    )


def required_x1(t: pa.Table) -> pa.Table:
    fields = [f.with_nullable(False) if f.name == "x1" else f for f in t.schema]
    return t.cast(pa.schema(fields))


def logical() -> pa.Table:
    return pa.table(
        {
            "d64": pa.array([Decimal("12.34"), Decimal("0.01")], pa.decimal128(10, 2)),
            "d32": pa.array([Decimal("-1.234"), Decimal("0.5")], pa.decimal128(5, 3)),
            "day": pa.array([datetime.date(1970, 1, 11), datetime.date(2000, 1, 1)], pa.date32()),
            "y": pa.array([1.0, 2.0], pa.float64()),
        }
    )


SHARDS = {
    "reg_0": (regression(0, 100), dict(row_group_size=50, use_dictionary=False)),
    "reg_1": (regression(100, 30), dict(use_dictionary=False)),
    "cls_0": (classification(0, 100), dict(row_group_size=50, use_dictionary=True, data_page_version="2.0")),
    "cls_1": (classification(100, 30), dict(use_dictionary=False, version="1.0")),
    "nulls": (regression(0, 10, x2_null=3), dict(use_dictionary=False)),
    "mismatch": (regression(0, 10, x2=pa.int64()), dict(use_dictionary=False)),
    "single": (regression(0, 10).select(["y"]), dict(use_dictionary=False)),
    "required": (required_x1(regression(0, 10)), dict(use_dictionary=False)),
    "x1_null": (regression(10, 10, x1_null=14), dict(use_dictionary=False)),
    "nested": (
        regression(0, 10).append_column("vals", pa.array([[i, i + 1] for i in range(10)], pa.list_(pa.int32()))),
        dict(use_dictionary=False),
    ),
    "logical": (logical(), dict(use_dictionary=False, store_decimal_as_integer=True)),
}

# Files written by a ParquetWriter closed without a table, which holds no
# row group.
EMPTY_SHARDS = {
    "no_rows": pa.schema([pa.field("x1", pa.float32()), pa.field("y", pa.float64())]),
}


def write(path: str, t: pa.Table, opts: dict) -> None:
    opts = dict(opts)
    row_group_size = opts.pop("row_group_size", None)
    pq.write_table(t, path, row_group_size=row_group_size, write_statistics=False, **opts)
    meta = pq.read_metadata(path)
    encodings = sorted({e for g in range(meta.num_row_groups) for c in range(meta.num_columns)
                        for e in meta.row_group(g).column(c).encodings})
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"{name}: {meta.num_rows} rows, {meta.num_row_groups} row groups, {os.path.getsize(path)} bytes, {encodings}")


def main() -> None:
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(root, "io", "parquet", "testdata"))
    p.add_argument("--shards", default=os.path.join(root, "data", "datasets", "testdata"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, opts in FILES.items():
        opts = dict(opts)
        write(os.path.join(args.out, name + ".parquet"), table(opts.pop("n")), opts)
    os.makedirs(args.shards, exist_ok=True)
    for name, (t, opts) in SHARDS.items():
        write(os.path.join(args.shards, name + ".parquet"), t, opts)
    for name, schema in EMPTY_SHARDS.items():
        path = os.path.join(args.shards, name + ".parquet")
        pq.ParquetWriter(path, schema, write_statistics=False).close()
        meta = pq.read_metadata(path)
        print(f"{name}: {meta.num_rows} rows, {meta.num_row_groups} row groups, {os.path.getsize(path)} bytes")


if __name__ == "__main__":
    main()
