import os,sys
from segmentation.constant import *
from segmentation.entity.config_entity import DataIngestionConfig
from segmentation.entity.artifact_entity import DataIngestionArtifact
from segmentation.data_access.mongo_data import get_collection_as_dataframe
from segmentation.utils.main_utils import read_yaml_file
from segmentation.logger.logger import logging
from segmentation.exception.exception import ApplicationException
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import yaml


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'>>'*30}Data Ingestion log started.{'<<'*30} \n\n")
            self.data_ingestion_config = data_ingestion_config
            
            file_path=SCHEMA_FILE_PATH
            schema_config=read_yaml_file(SCHEMA_FILE_PATH)
            
            self.file_name=schema_config['FileName']
            
            logging.info(f" Data File Label : {self.file_name} ")
            # Data file label 
            

        except Exception as e:
            raise ApplicationException(e,sys) from e
    
    
    def get_data_from_mongo(self):
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            data_base_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name
            
            logging.info(f"Database : {data_base_name} ")
            logging.info(f"Collection : {collection_name} ")
            #Exporting collection data as pandas dataframe
            df:pd.DataFrame  = get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name, 
                collection_name=self.data_ingestion_config.collection_name)
            
            logging.info(df.head(10))

            logging.info("Saving Data from Database to local folder ....")
            
            # Raw Data Directory Path
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            
            logging.info(f" Raw Data directory : {raw_data_dir}")

            # Make Raw data Directory
            os.makedirs(raw_data_dir, exist_ok=True)
            
            csv_file_name=self.file_name
            raw_file_path = os.path.join(raw_data_dir, csv_file_name)
            df.to_csv(raw_file_path)
            
        
            # copy the the extracted csv from raw_data_dir ---> ingested Data 
            ingest_directory=os.path.join(self.data_ingestion_config.ingested_data_dir)
            os.makedirs(ingest_directory,exist_ok=True)

            # Updating file name 
            ingest_file_path = os.path.join(self.data_ingestion_config.ingested_data_dir, csv_file_name)
            
            # Copy the extracted CSV file
            shutil.copy2(raw_file_path, ingest_file_path)
            
            logging.info(" Data stored in ingested Directory ")
            
            
            return ingest_file_path
            

        except Exception as e:
            raise ApplicationException(e, sys) from e 
        
        
        
    def train_data(self,csv_file_path):
        
                    
        train_file_path=self.data_ingestion_config.train_file_path
        
        os.makedirs(train_file_path,exist_ok=True)
        
        # Load data from the CSV file
        train_data = pd.read_csv(csv_file_path)
        
        # Drop unnamed column if it exists
        if 'Unnamed: 0' in train_data.columns:
            train_data.drop(columns=['Unnamed: 0'], inplace=True)

        # Save the training and testing data into separate CSV files
        train_file_path=os.path.join(train_file_path,self.file_name)

        
        logging.info(f" Train File path : {train_file_path}")
        
        train_data.to_csv(train_file_path, index=False)
        
        logging.info("---Data Split Done ---")
        logging.info(f"Shape of the data Train Data : {train_data.shape}")
        
        logging.info(f" Train File path : {train_file_path}")
        
        
        data_ingestion_artifact=DataIngestionArtifact(train_file_path=train_file_path)
        
        return data_ingestion_artifact



    def initiate_data_ingestion(self):
        try:
            
            logging.info("---------------------- Initiating Data Ingestion -----------------")
            file_path='params.yaml'
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"The file '{file_path}' has been deleted.")
            else:
                print(f"The file '{file_path}' does not exist.")
                        
                        
            
            logging.info("Downloading data from mongo ")
             # Define the path where you want to save the YAML file
            file_path =ARTIFACT_ENTITY_YAML_FILE_PATH
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            artifact_data={}

            # Write data to YAML file
            with open(file_path, 'w') as yaml_file:
                yaml.dump(artifact_data, yaml_file, default_flow_style=False)

            print(f"Artifact YAML file  has been created successfully.")
            ingest_file_path=self.get_data_from_mongo()
            
            logging.info("Splitting data .... ")
            
            return  self.train_data(csv_file_path=ingest_file_path)


        except Exception as e:
            raise ApplicationException(e,sys) from e