import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pyarrow_pandas_converter import PyArrowTableFlattener, to_pandas_safe


def test_flatten_table_with_composite_keys():

    test_table = pa.table(
        [
            pa.array([1, 2]),
            pa.array([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]),
            pa.array(
                [
                    {"student": {"name": "Alice", "age": 30}},
                    {"student": {"name": "Bob", "age": 25}},
                ]
            ),
            pa.array([[1], [5]]),
        ],
        names=["id", "person", "complicated_person", "simple_list"],
    )

    flattener = PyArrowTableFlattener(table=test_table)
    flat_table = flattener.flatten_all_columns(
        recursive=True,
        composite_names=True,
        keep_nested_columns=False,
    )

    assert flat_table["person.age"] == pa.chunked_array([[30, 25]])
    assert flat_table["person.name"] == pa.chunked_array([["Alice", "Bob"]])
    assert flat_table["complicated_person.student.age"] == pa.chunked_array([[30, 25]])
    assert flat_table["complicated_person.student.name"] == pa.chunked_array(
        [["Alice", "Bob"]]
    )
    assert flat_table["simple_list"] == pa.chunked_array([[1, 5]])


def test_flatten_structs_only():

    test_table = pa.table(
        [
            pa.array([[1], [5]]),
            pa.array([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]),
        ],
        names=["simple_list", "person"],
    )

    flattener = PyArrowTableFlattener(table=test_table)

    not_flat_table = flattener.flatten_struct_columns(
        recursive=True,
    )

    assert not_flat_table["person.age"] == pa.chunked_array([[30, 25]])
    assert not_flat_table["person.name"] == pa.chunked_array([["Alice", "Bob"]])
    assert not_flat_table["simple_list"] == pa.chunked_array([[[1], [5]]])


def test_flatten_list_column():

    test_table = pa.table([pa.array([[1], [5]])], names=["simple_list"])

    flattener = PyArrowTableFlattener(table=test_table)

    flat_table = flattener.flatten_all_columns(
        recursive=True,
        composite_names=True,
        keep_nested_columns=False,
    )

    assert flat_table["simple_list"] == pa.chunked_array([[1, 5]])


def test_safe_conversion_to_pandas():

    test_table = pa.table([pa.array([[1], [None], [5]])], names=["list_with_none"])

    flattener = PyArrowTableFlattener(table=test_table)

    flat_table = flattener.flatten_all_columns(
        recursive=True,
    )

    df = to_pandas_safe(flat_table)

    assert pd.isna(df["list_with_none"][1])
    assert len(df["list_with_none"]) == 3


def test_write_read_parquet():

    test_table = pa.table([pa.array([[1], [None], [5]])], names=["list_with_none"])

    flattener = PyArrowTableFlattener(table=test_table)

    flat_table = flattener.flatten_all_columns(
        recursive=True,
    )

    test_file = "test.parquet"
    pq.write_table(flat_table, test_file)
    df = pd.read_parquet(test_file)
    os.remove(test_file)

    assert pd.isna(df["list_with_none"][1])
    assert len(df["list_with_none"]) == 3


def test_flatten_map():
    data = [[('x', 1), ('y', 0)], [('a', 2), ('b', 45)]]
    ty = pa.map_(pa.string(), pa.int64())
    map_array = pa.array(data, type=ty)

    test_table = pa.table([map_array], names=["map_array"])

    flattener = PyArrowTableFlattener(table=test_table)

    flat_table = flattener.flatten_all_columns(
        recursive=True,
        composite_names=True,
        keep_nested_columns=False,
    )

    assert flat_table["map_array.keys"] == pa.chunked_array([["x", "y", "a", "b"]])
    assert flat_table["map_array.values"] == pa.chunked_array([[1, 0, 2, 45]])
