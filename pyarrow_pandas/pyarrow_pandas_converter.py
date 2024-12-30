"""
Module to help in conversion from pyarrow table
int pandas dataframes for cases like columns with nested types
and nullable values

"""

import logging
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


class PyArrowTableFlattener:
    """
    Class responsible for flattening nested columns in a PyArrow table.
    """

    def __init__(self, table: Optional[pa.Table] = None):
        """
        Initializes the PyArrowTableFlattener with an optional PyArrow table.

        Args:
            table (Optional[pa.Table]): The PyArrow table to be flattened, defaults to None.
        """
        self._table = table

    def set_table(self, table: pa.Table):
        """
        Sets the PyArrow table to be flattened.

        Args:
            table (pa.Table): The PyArrow table to set.
        """
        self._table = table

    def flatten_all_columns(
        self,
        recursive: bool = False,
        composite_names: bool = False,
        keep_nested_columns: bool = False,
    ) -> pa.Table:
        """
        Flattens all nested columns in the table.

        Args:
            recursive (bool): If True, will recursively flatten nested columns. Defaults to False.
            composite_names (bool): If True, will use composite names for nested fields. Defaults to False.
            keep_nested_columns (bool): If True, will keep the original nested columns. Defaults to False.

        Returns:
            pa.Table: The flattened PyArrow table.
        """
        self._validate_table_set()
        while self._has_nested_columns():
            logger.info(f"Flattening table with {len(self._table.columns)} columns.")
            self._table = self._flatten_nested_columns(
                composite_names, keep_nested_columns
            )
            if not recursive:
                break
        return self._table

    def flatten_struct_columns(self, recursive: bool = False) -> pa.Table:
        """
        Flattens struct columns in the table.

        Args:
            recursive (bool): If True, will recursively flatten struct columns. Defaults to False.

        Returns:
            pa.Table: The flattened PyArrow table with struct columns flattened.
        """
        self._validate_table_set()
        while self._has_struct_columns():
            logger.info(
                f"Flattening struct columns in table with {len(self._table.columns)} columns."
            )
            self._table = self._table.flatten()
            if not recursive:
                break
        return self._table

    ######## PRIVATE METHODS #########

    def _validate_table_set(self):
        """
        Validates that a PyArrow table has been set for flattening.

        Raises:
            ValueError: If no table has been set.
        """
        if self._table is None:
            logger.error("No table set for flattening.")
            raise ValueError("Table must be set before flattening.")

    def _has_struct_columns(self) -> bool:
        """
        Checks if the table has any struct columns.

        Returns:
            bool: True if there are struct columns, False otherwise.
        """
        return any(pa.types.is_struct(field.type) for field in self._table.schema)

    def _has_nested_columns(self) -> bool:
        """
        Checks if the table has any nested columns.

        Returns:
            bool: True if there are nested columns, False otherwise.
        """
        return any(pa.types.is_nested(field.type) for field in self._table.schema)

    def _flatten_nested_columns(
        self, composite_names: bool, keep_nested_columns: bool
    ) -> pa.Table:
        """
        Flattens the nested columns in the table.

        Args:
            composite_names (bool): If True, will use composite names for nested fields.
            keep_nested_columns (bool): If True, will keep the original nested columns.

        Returns:
            pa.Table: The flattened PyArrow table.
        """
        start_time = time.perf_counter()
        arrays_dict = self._build_flattened_arrays_dict(
            composite_names, keep_nested_columns
        )
        flattened_table = pa.table(arrays_dict)
        self._log_flattening_performance(start_time, flattened_table)
        return flattened_table

    def _build_flattened_arrays_dict(
        self, composite_names: bool, keep_nested_columns: bool
    ) -> Dict[str, pa.Array]:
        """
        Builds a dictionary of flattened arrays from the nested columns.

        Args:
            composite_names (bool): If True, will use composite names for nested fields.
            keep_nested_columns (bool): If True, will keep the original nested columns.

        Returns:
            Dict[str, pa.Array]: A dictionary mapping column names to their corresponding flattened arrays.
        """
        arrays_dict = {}
        for field in self._table.schema:
            if not pa.types.is_nested(field.type) or keep_nested_columns:
                arrays_dict[field.name] = self._table.column(field.name)
            if pa.types.is_struct(field.type):
                arrays_dict.update(self._flatten_struct_field(field, composite_names))
            elif pa.types.is_list(field.type):
                arrays_dict.update(self._flatten_list_field(field))
            elif pa.types.is_map(field.type):
                arrays_dict.update(self._flatten_map_field(field, composite_names))
        return arrays_dict

    def _flatten_struct_field(
        self, field: pa.Field, composite_names: bool
    ) -> Dict[str, pa.Array]:
        """
        Flattens a struct field into separate columns.

        Args:
            field (pa.Field): The struct field to flatten.
            composite_names (bool): If True, will use composite names for subfields.

        Returns:
            Dict[str, pa.Array]: A dictionary mapping subfield names to their corresponding arrays.
        """
        subfield_names = self._generate_subfield_names(field, composite_names)
        flattened_columns = self._table.column(field.name).flatten()
        return dict(zip(subfield_names, flattened_columns))

    def _flatten_list_field(self, field: pa.Field) -> Dict[str, pa.Array]:
        """
        Flattens a list field into a single array.

        Args:
            field (pa.Field): The list field to flatten.

        Returns:
            Dict[str, pa.Array]: A dictionary mapping the field name to the flattened array.
        """
        flattened_column = pc.list_flatten(self._table.column(field.name))
        return {field.name: flattened_column}

    def _flatten_map_field(
        self, field: pa.Field, composite_names: bool
    ) -> Dict[str, pa.Array]:
        """
        Flattens a map_array field into separate columns.

        Args:
            field (pa.Field): The map_array field to flatten.
            composite_names (bool): If True, will use composite names for subfields.

        Returns:
            Dict[str, pa.Array]: A dictionary mapping subfield names to their corresponding arrays.
        """
        keys = []
        values = []

        map_array = self._table.column(field.name)

        for i in range(len(map_array)):
            if map_array[i] is not None:
                for j in range(len(map_array[i])):
                    keys.append(map_array[i][j][0])  # Get the key
                    values.append(map_array[i][j][1])  # Get the value

        name_prefix = f"{field.name}." if composite_names else ""

        subfield_names = [f"{name_prefix}keys", f"{name_prefix}values"]

        flattened_columns = [pa.array(keys), pa.array(values)]

        return dict(zip(subfield_names, flattened_columns))

    @staticmethod
    def _generate_subfield_names(field: pa.Field, composite_names: bool) -> List[str]:
        """
        Generates names for subfields in a struct field.

        Args:
            field (pa.Field): The struct field containing subfields.
            composite_names (bool): If True, will generate composite names.

        Returns:
            List[str]: A list of generated subfield names.
        """
        return (
            [f"{field.name}.{subfield.name}" for subfield in field.type]
            if composite_names
            else [subfield.name for subfield in field.type]
        )

    @staticmethod
    def _log_flattening_performance(start_time: float, flattened_table: pa.Table):
        """
        Logs the performance of the flattening operation.

        Args:
            start_time (float): The time at which flattening started.
            flattened_table (pa.Table): The resulting flattened PyArrow table.
        """
        elapsed_time = time.perf_counter() - start_time
        nbytes = flattened_table.nbytes
        logger.info(
            f"Time to flatten table: {elapsed_time:.4f} seconds. Size: {nbytes} bytes."
        )


# Convert to Pandas DataFrame, expanding the map column
def map_to_dict(map_array: pa.Array) -> List[Dict]:
    """
    Converts a PyArrow map array to a list of dictionaries.

    Args:
        map_array (pa.Array): The PyArrow map array to convert.

    Returns:
        List[Dict]: A list of dictionaries representing the map array.
    """
    return [dict(zip(m.keys(), m.values())) for m in map_array]


def to_pandas_safe(table: pa.Table) -> pd.DataFrame:
    """
    Converts a PyArrow table to a Pandas DataFrame with explicit type mapping.

    Args:
        table (pa.Table): The PyArrow table to convert.

    Returns:
        pd.DataFrame: The converted Pandas DataFrame.
    """
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
        pa.map_: map_to_dict,
    }

    return table.to_pandas(types_mapper=dtype_mapping.get)
