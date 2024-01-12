import pandas as pd
import json
import os
from src.segmentation.data_access.mongo_access import mongo_client
from data_schema import write_schema_yaml
from src.segmentation.utils.main_utils import read_yaml_file

# Extracting project config 
config_file_path = os.path.join(os.getcwd(), 'config', 'config.yaml')
project_config = read_yaml_file(config_file_path)

# Accessing file Label 
data_file_label = project_config['data_file_label']

# Data Ingestion config 
data_ingestion_config = project_config['data_ingestion_config']
data_base = data_ingestion_config['data_base']
collection_name = data_ingestion_config['collection_name']

# Accessing Mongo client for database access
client = mongo_client()
DATA_FILE_PATH = os.path.join(os.getcwd(), 'data', data_file_label)
DATABASE_NAME = data_base
COLLECTION_NAME = collection_name

if __name__ == "__main__":
    

    
    # Read CSV file into a DataFrame
    df = pd.read_csv(DATA_FILE_PATH)
    
    # Drop unnecessary columns
    df.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Reset index
    df.reset_index(drop=True, inplace=True)
    # Write schema YAML
    write_schema_yaml(dataframe=df,csv_file=DATA_FILE_PATH)
    
    # Convert DataFrame to a list of dictionaries
    json_records = json.loads(df.to_json(orient='records'))
    
    # Delete existing data in the collection
    client[DATABASE_NAME][COLLECTION_NAME].delete_many({})
    
    # Insert new data into the collection
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_records)
    
    print(f"Rows and columns: {df.shape}")
    print("Data inserted successfully.")
