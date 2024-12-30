# Pyarrow-Pandas

Converter from pyarrow Tables into Pandas dataframes for nested objects


## Problem statement

When converting between PyArrow and Pandas DataFrames, you might encounter a few common issues

### Type Differences:

Nullable columns:

Pandas doesn't natively support nullable columns of arbitrary types, while PyArrow does.
This can lead to data loss or conversion errors if your PyArrow table contains nulls in columns that Pandas can't handle.


Datetime resolution:
Pandas' datetime64 type is fixed to nanosecond resolution, while PyArrow allows for different resolutions.
This can cause mismatches when converting date/time data.

Unsupported types:
PyArrow might not support all Pandas types, and vice versa.
For example, Pandas' categorical type is not directly supported by PyArrow.


### Nested Data:
PyArrow Tables can represent nested data structures (e.g., structs, lists, maps), which Pandas DataFrames cannot.
This means you may lose hierarchical information when converting from PyArrow to Pandas.

Pandas DataFrames can contain columns with complex objects (e.g., lists, dictionaries), which PyArrow may not fully support.
This can lead to conversion errors or data loss.

### Performance:
While PyArrow is generally faster than Pandas for many operations,
converting between the two can sometimes introduce overhead.
This is especially true for large datasets or complex data structures.


## Solutions:

### Use explicit conversions:
You can use the `to_pandas()` method on a PyArrow Table and the `Table.from_pandas()` function to convert between formats.


### Handle nulls carefully:
The `pyarrow.Table.to_pandas()` method has a types_mapper keyword that can be used to override the default data type used for the resulting pandas DataFrame.
This way, you can instruct Arrow to create a pandas DataFrame using nullable dtypes.

The types_mapper keyword expects a function that will return the pandas data type to use given a pyarrow data type.
By using the `dict.get` method, we can create such a function using a dictionary.

If you want to use all currently supported nullable dtypes by pandas, this dictionary becomes:


```
dtype_mapping = {
    pa.int8(): pd.Int8Dtype(),
    pa.int16(): pd.Int16Dtype(),
    pa.int32(): pd.Int32Dtype(),
    pa.int64(): pd.Int64Dtype(),
    pa.uint8(): pd.UInt8Dtype(),
    pa.uint16(): pd.UInt16Dtype(),
    pa.uint32(): pd.UInt32Dtype(),
    pa.uint64(): pd.UInt64Dtype(),
    pa.bool_(): pd.BooleanDtype(),
    pa.float32(): pd.Float32Dtype(),
    pa.float64(): pd.Float64Dtype(),
    pa.string(): pd.StringDtype(),
}

df = table.to_pandas(types_mapper=dtype_mapping.get)
```

Consider using `fillna()` in Pandas to handle null values before converting to PyArrow.


### Flatten nested data:

Converting nested objects from a PyArrow Table to a Pandas DataFrame can be achieved in a few ways, depending on your desired output and the structure of your nested data.
Here's a breakdown of the methods:


Using `to_pandas()` with `table.flatten()`

This method flattens the nested structure, creating new columns for each nested field.

```
import pyarrow as pa
import pandas as pd

# Create a PyArrow Table with nested data
table = pa.table({
    'id': [1, 2],
    'person': [
        {'name': 'Alice', 'age': 30},
        {'name': 'Bob', 'age': 25}
    ]
})

# Convert the table to a Pandas DataFrame, flattening the nested structure
df = table.flatten().to_pandas()
print(df)

```


Using json_normalize() for complex nesting


If you have more complex nested structures, you can use json_normalize() from the `pandas` library.

```
import pyarrow as pa
import pandas as pd
from pandas.io.json import json_normalize

# Create a PyArrow Table with nested data
table = pa.table({
    'id': [1, 2],
    'person': [
        {'name': 'Alice', 'address': {'city': 'New York'}},
        {'name': 'Bob', 'address': {'city': 'Los Angeles'}}
    ]
})

# Convert the table to a Pandas DataFrame
df = table.to_pandas()

# Normalize the nested 'person' column
df = json_normalize(df['person'])
print(df)

```


Custom flattening for specific needs

If you need more control over the flattening process, you can write custom code to handle the nested objects.


```
import pyarrow as pa
import pandas as pd

# Create a PyArrow Table with nested data
table = pa.table({
    'id': [1, 2],
    'person': [
        {'name': 'Alice', 'age': 30},
        {'name': 'Bob', 'age': 25}
    ]
})

# Convert the table to a Pandas DataFrame
df = table.to_pandas()

# Custom flattening
df['name'] = df['person'].apply(lambda x: x['name'])
df['age'] = df['person'].apply(lambda x: x['age'])
df.drop('person', axis=1, inplace=True)
print(df)
```



### Performance

PyArrow to Pandas conversion can be optimized using the `to_pandas()` method's arguments,
such as `self_destruct` and `split_blocks`.


## Additional links


https://github.com/apache/arrow/pull/44720

https://github.com/pandas-dev/pandas/issues/53011



## PyArrow Table Flattener

### Overview

The `PyArrow Table Flattener` is a Python module designed to facilitate the conversion of PyArrow tables into Pandas DataFrames, particularly for use cases involving columns with nested types and nullable values. This module provides functionality to flatten nested structures within PyArrow tables, making it easier to work with complex data types in Pandas.

### Features

- **Flatten Nested Columns**: The module can flatten nested columns in a PyArrow table, allowing for easier manipulation and analysis of data.
- **Recursive Flattening**: Supports recursive flattening of nested structures, ensuring all levels of nested columns are handled.
- **Composite Naming**: Option to create composite names for new columns generated during flattening, enhancing clarity and usability.
- **Type Safety**: Converts PyArrow tables to Pandas DataFrames with explicit type mapping to ensure data integrity.

### Installation

To use the `PyArrow Table Flattener`, ensure you have the following dependencies installed:

```bash
pip install pandas pyarrow
```

### Usage

1. **Import the Module**: Start by importing the necessary classes and functions.

```bash
import pyarrow as pa
from your_module_name import PyArrowTableFlattener, to_pandas_safe
```

2. **Create a PyArrow Table**: Create or load a PyArrow table that you wish to flatten.

```bash
# Example of creating a PyArrow table
data = {
  'id': [1, 2],
  'info': pa.table({'name': ['Alice', 'Bob'], 'age': [25, 30]}),
  'scores': pa.array([[85, 90], [78, 88]])
}
table = pa.table(data)
```

3. **Flatten the Table**: Instantiate the `PyArrowTableFlattener` class and use the `flatten_all` method to flatten the table.

 ```bash
 flattener = PyArrowTableFlattener(table)
 flattened_table = flattener.flatten_all(recursive=True, composite_names=True)
 ```

4. **Convert to Pandas DataFrame**: Use the `to_pandas_safe` function to convert the flattened PyArrow table to a Pandas DataFrame.

```bash
df = to_pandas_safe(flattened_table)
```

### Method Descriptions

- **`flatten_all(recursive: bool = False, composite_names: bool = False, keep_nested_columns: bool = False) -> pa.Table`**: Flattens all nested columns in the table based on the provided options.

- **`flatten_structs(recursive: bool = False) -> pa.Table`**: Flattens only struct columns in the table, using a built-in method for simplicity.

- **`to_pandas_safe(table: pa.Table) -> pd.DataFrame`**: Converts a PyArrow table to a Pandas DataFrame with explicit type mapping to ensure compatibility.

### Logging

The module uses Python's built-in logging library to provide informative logs during the flattening process. The logs include the number of columns being flattened and performance metrics such as execution time and size of the resulting table.

### Example

Here’s a complete example demonstrating the usage of the module:

```bash
import pyarrow as pa
from your_module_name import PyArrowTableFlattener, to_pandas_safe

# Create a sample PyArrow table
data = {
    'id': [1, 2],
    'info': pa.table({'name': ['Alice', 'Bob'], 'age': [25, 30]}),
    'scores': pa.array([[85, 90], [78, 88]])
}
table = pa.table(data)

# Flatten the table
flattener = PyArrowTableFlattener(table)
flattened_table = flattener.flatten_all(recursive=True, composite_names=True)

# Convert to Pandas DataFrame
df = to_pandas_safe(flattened_table)

print(df)
```

### Conclusion

The `PyArrow Table Flattener` module is a powerful tool for data scientists and engineers working with complex data structures in PyArrow. By providing easy-to-use methods for flattening nested columns and converting to Pandas DataFrames, it streamlines data processing workflows and enhances data accessibility.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
