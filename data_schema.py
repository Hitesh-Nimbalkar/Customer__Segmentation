import yaml
import pandas as pd
import os
def write_schema_yaml(dataframe, csv_file):
    # Get number of columns and column names
    num_cols = len(dataframe.columns)
    column_names = dataframe.columns.tolist()
    column_dtypes = dataframe.dtypes.astype(str).tolist()  # Convert data types to string for YAML compatibility

    # Create schema dictionary
    schema = {
        "FileName": os.path.basename(csv_file),
        "NumberOfColumns": num_cols,
        "ColumnNames": dict(zip(column_names, column_dtypes))
    }

    # Write schema to schema.yaml file
    ROOT_DIR = os.getcwd()
    SCHEMA_PATH = os.path.join(ROOT_DIR, 'config', 'schema.yaml')

    with open(SCHEMA_PATH, "w") as file:
        yaml.dump(schema, file)