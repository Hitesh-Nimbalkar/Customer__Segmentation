import os 
import sys
import pandas as pd
import numpy as np
import scipy
from segmentation.logger.logger import logging
from segmentation.exception.exception import ApplicationException
from segmentation.entity.artifact_entity import *
from segmentation.entity.config_entity import *
from segmentation.utils.main_utils import *
from segmentation.constant import *
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self,drop_columns):
        
        """
        This class applies necessary Feature Engneering 
        """
        logging.info(f"\n{'*'*20} Feature Engneering Started {'*'*20}\n\n")
        

                                ############### Accesssing Column Labels #########################
                                
                                
                 #   Schema.yaml -----> Data Tranformation ----> Method: Feat Eng Pipeline ---> Class : Feature Eng Pipeline              #
                                
                                

        self.columns_to_drop = drop_columns

        
                                ########################################################################
        
        logging.info(f" Numerical Columns , Categorical Columns , Target Column initialised in Feature engineering Pipeline ")


    # Feature Engineering Pipeline 
    
    
    
    
                                                    ######################### Data Modification ############################

    def drop_columns(self,X:pd.DataFrame):
        try:
            columns=X.columns
            
            logging.info(f"Columns before drop  {columns}")
            
            # Columns Dropping
            drop_column_labels=self.columns_to_drop
            
            logging.info(f" Dropping Columns {drop_column_labels} ")
            
            X=X.drop(columns=drop_column_labels,axis=1)
            
            return X
        
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
        
    def drop_rows_with_nan(self, X: pd.DataFrame):
        # Log the shape before dropping NaN values
        logging.info(f"Shape before dropping NaN values: {X.shape}")
        
        # Drop rows with NaN values
        X = X.dropna()
        #X.to_csv("Nan_values_removed.csv", index=False)
        
        # Log the shape after dropping NaN values
        logging.info(f"Shape after dropping NaN values: {X.shape}")
        
        logging.info("Dropped NaN values.")
        
        return X
 
    def drop_duplicates(self,X:pd.DataFrame):
        """
        Drops duplicate rows from a pandas DataFrame and returns the modified DataFrame.
        
        Args:
            df (pandas.DataFrame): The DataFrame to remove duplicate rows from.
            
        Returns:
            pandas.DataFrame: The modified DataFrame with duplicate rows removed.
        """
        
        print(" Drop duplicate value")
        X = X.drop_duplicates()
        
        
        return X
   
    def remove_duplicate_rows_keep_last(self,X):
        
        logging.info(f"DataFrame shape before removing duplicates: {X.shape}")
        num_before = len(X)
        X.drop_duplicates(inplace = True)
        num_after = len(X)
        
        num_duplicates = num_before - num_after
        logging.info(f"Removed {num_duplicates} duplicate rows")
        logging.info(f"DataFrame shape after removing duplicates: {X.shape}")
        
        return X


    def convert_nan_null_to_nan(self,X:pd.DataFrame):
        # Convert "NAN" and "NULL" values to np.nan
        X.replace(["NAN", "NULL","nan"], np.nan, inplace=True)

        # Return the updated DataFrame
        return X

    def day_since_enrollment(self,data):
        # Convert 'Dt_Customer' to a date-time format
        data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')

        # Find the max and min dates in the Dt_Customer column
        max_date = data['Dt_Customer'].max()
        min_date = data['Dt_Customer'].min()
        
        
        logging.info(f"The newest customer's enrolment date in the records:", max_date)
        logging.info("The oldest customer's enrolment date in the records:", min_date)

        # Calculate the number of days between the enrolment date and the maximum date
        data['Days_since_enrollment'] = (max_date - data['Dt_Customer']).dt.days

        return data

    
    
    def add_columns(self,df, column_names, new_column_name):
        df[new_column_name] = df[column_names].sum(axis=1)
        return df

    
    def rename_columns(self,dataframe):
        # Renaming the specified columns
        column_mapping = {'MntWines': 'Wines',
                        'MntFruits': 'Fruits',
                        'MntMeatProducts': 'Meat',
                        'MntFishProducts': 'Fish',
                        'MntSweetProducts': 'Sweet',
                        'MntGoldProds': 'Gold',
                        'NumDealsPurchases': 'DealsPurchases',
                        'NumWebPurchases': 'Web',
                        'NumCatalogPurchases': 'Catalog',
                        'NumStorePurchases': 'Store',
                        'NumWebVisitsMonth': 'WebVisitsMonth'}

        dataframe.rename(columns=column_mapping, inplace=True)
        return dataframe

    
        
    def run_data_modification(self,data):
        
        X=data.copy()
        
        # Drop na from dataframe
        X.dropna(inplace=True)
    
        # Removing duplicated rows 
        X=self.remove_duplicate_rows_keep_last(X)

        # make Null as np.nan
        X=self.convert_nan_null_to_nan(X)
        
        # Drop rows with nan
        X=self.drop_rows_with_nan(X)
        
        # Consumer Data 
        logging.info(" ------ Modifying consumer Data ------")
        # Total Offspring
        logging.info(" Creating Total Offspring column")
        X['Total_Offsprings'] = X['Kidhome'] + X['Teenhome']

        # Living With
        # Deriving Living attributes based on the marital status
        logging.info(" Creating Living_With column")
        X['Living_With'] = X['Marital_Status'].replace({'Married': 'Partner', 'Together': 'Partner', 'Single': 'Alone',
                                                        'Divorced': 'Divorced', 'Widow': 'Widow', 'Absurd': 'Alone',
                                                        'YOLO': 'Alone'})

        # Feature indicating Family size
        logging.info(" Creating Family_Size column")
        X['Family_Size'] = (X['Living_With'].map({'Partner': 2, 'Alone': 1, 'Divorced': 1, 'Widow': 1}).astype(int) + X['Total_Offsprings']).astype(int)

        
       
        # Education column
        logging.info(" Modifying  Education column")
        X['Education'] = X['Education'].replace({'Basic': 'Undergraduate',
                                                '2n Cycle': 'Undergraduate',
                                                'Graduation': 'Graduate',
                                                'Master': 'Postgraduate',
                                                'PhD': 'Postgraduate'})

        # Creating column Day since enrollment
   #     logging.info(" Creating Day_since_enrollment column")
   #     X = self.day_since_enrollment(X)

        # Creating Age column
        # Calculate the age of customers
        logging.info(" Creating Age column")
        X['Age'] = 2023 - X['Year_Birth']
        
        
        logging.info(" ------ Consumer Data Modified -------- ")
        
        # Product Data 
        column_names = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        X = self.add_columns(X,new_column_name="Total_Amount",column_names=column_names)
        
        # Frequency 
        columns=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3','AcceptedCmp4', 'AcceptedCmp5', 'Response']
        X=self.add_columns(X,columns,new_column_name='Frequency')
        
        # Place dataframe 
        X=self.add_columns(X,new_column_name='Total_purchases',column_names=['NumWebPurchases','NumCatalogPurchases','NumStorePurchases'])
        
        # Rename columns for clarity and ease of use
        X =self.rename_columns(dataframe=X)
        
        # Promotion Data 
        X['AcceptedCmp1'] = X['AcceptedCmp1'].replace(1, 1)
        X['AcceptedCmp2'] = X['AcceptedCmp2'].replace(1, 2)
        X['AcceptedCmp3'] = X['AcceptedCmp3'].replace(1, 3)
        X['AcceptedCmp4'] = X['AcceptedCmp4'].replace(1, 4)
        X['AcceptedCmp5'] = X['AcceptedCmp5'].replace(1, 5)
        X['Response'] = X['Response'].replace(1, 6)
    
        # create new feature for the ratio of online purchases to total purchases
        X['online_purchase_ratio'] = X['Web'] / (X['Web'] + X['Catalog'] + X['Store'])
        # Drop Columns 
        X=self.drop_columns(X=X)
        
        logging.info(f" Columne after dropping : {X.columns}")
        
        
        return X
    
    
    

                                            ######################### Outiers ############################
    

    
    def outlier(self,X):
        
        X = X[(X['Income'] < 600000)]
        return X
    
    
    def data_wrangling(self,X:pd.DataFrame):
        try:

            
            # Data Modification 
            data_modified=self.run_data_modification(data=X)
            
            logging.info(" Data Modification Done")
            
            # Removing outliers 
            
            logging.info(" Removing Outliers")
            
            df_outlier_removed=self.outlier(X=data_modified)
        
            
            
            return df_outlier_removed
    
        
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
        
    
    
    def fit(self,X,y=None):
        return self
    
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            data_modified=self.data_wrangling(X)

            
            logging.info(f"Original Data  : {X.shape}")
            logging.info(f"Shape Modified Data : {data_modified.shape}")
         
          #  data_modified.to_csv("data_modified.csv",index=False)
            logging.info(" Data Wrangaling Done ")

            return data_modified
        except Exception as e:
            raise ApplicationException(e,sys) from e


class DataProcessor:
    def __init__(self, numerical_cols, categorical_cols):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols

        # Define preprocessing steps using a Pipeline
        categorical_transformer = Pipeline(
            steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]
        )
        numerical_transformer = Pipeline(
            steps=[
                ('log_transform', FunctionTransformer(np.log1p, validate=False)),
                ('scaler', StandardScaler())  # Add StandardScaler for numerical columns
            ]
        )

        # Create a ColumnTransformer to apply transformations
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.categorical_cols),
                ('num', numerical_transformer, self.numerical_cols)
            ],
            remainder='passthrough'
        )

    def get_preprocessor(self):
        return self.preprocessor

    def fit_transform(self, data):
        # Fit and transform the data using the preprocessor
        transformed_data = self.preprocessor.fit_transform(data)

        # Convert the sparse matrix to a dense array if necessary
        if isinstance(transformed_data, (scipy.sparse.csr.csr_matrix, scipy.sparse.csc.csc_matrix)):
            transformed_data = transformed_data.toarray()

        return transformed_data

    def transform(self, data):
        # Transform the data using the preprocessor (assuming it has been fit already)
        transformed_data = self.preprocessor.transform(data)

        # Convert the sparse matrix to a dense array if necessary
        if isinstance(transformed_data, (scipy.sparse.csr.csr_matrix, scipy.sparse.csc.csc_matrix)):
            transformed_data = transformed_data.toarray()

        return transformed_data



class DataTransformation:
    
    
    def __init__(self, data_transformation_config: DataTransformationConfig,
                    data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"\n{'*'*20} Data Transformation log started {'*'*20}\n\n")
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            
                                ############### Accesssing Column Labels #########################
            
            # Transformation Yaml File path 
            self.transformation_yaml = read_yaml_file(file_path=TRANSFORMATION_YAML_FILE_PATH)

            self.drop_columns=self.transformation_yaml[DROP_COLUMNS]
          
            
                                ########################################################################
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
    def get_feature_engineering_object(self):
        try:
            
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering(  ))])
            return feature_engineering
        except Exception as e:
            raise ApplicationException(e,sys) from e
    def separate_numerical_categorical_columns(self,df):
        numerical_columns = []
        categorical_columns = []

        for column in df.columns:
            if df[column].dtype == 'int64' or df[column].dtype == 'float64':
                numerical_columns.append(column)
            else:
                categorical_columns.append(column)

        return numerical_columns, categorical_columns



    def get_feature_engineering_object(self):
        try:
            
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering(drop_columns=self.drop_columns))])
            return feature_engineering
        except Exception as e:
            raise ApplicationException(e,sys) from e
   




    def initiate_data_transformation(self):
        try:
            # Data validation Artifact ------>Accessing train and test files 
            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_validation_artifact.validated_train_path
      

            train_df = pd.read_csv(train_file_path)
          
            logging.info(f" Accessing train file from :{train_file_path}")    
    
            logging.info(f"Train Data  :{train_df.shape}")
             # Transforming Data 
            numerical_cols,categorical_cols=self.separate_numerical_categorical_columns(df=train_df)
        
            
            logging.info(f"Numerical Column :{numerical_cols}")
            logging.info(f"Categorical Column :{categorical_cols}")

            col = numerical_cols + categorical_cols
            # All columns 
            logging.info("All columns: {}".format(col))
            
                        
            # Feature Engineering 
            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()
            
            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            logging.info(">>>" * 20 + " Training data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Train Data ")
            train_df = fe_obj.fit_transform(train_df)

            #logging.info(f" Columns in feature enginering {feature_eng_test_df.columns}")
            logging.info(f"Saving feature engineered training  dataframe.")
            transformed_train_dir = self.data_transformation_config.transformed_train_dir

            Feature_eng_train_file_path = os.path.join(transformed_train_dir,"Feature_engineering.csv")
            
            save_data(file_path = Feature_eng_train_file_path, data = train_df)
            
            # Train Data 
            logging.info(f"Feature Engineering of train and test Completed.")
            logging.info (f"  Shape of Featured Engineered Data Train Data : {train_df.shape} ")
            
            feature_eng_train_df:pd.DataFrame = train_df.copy()
          #  feature_eng_train_df.to_csv("feature_eng_train_df.csv")
            logging.info(f" Columns in feature enginering Train {feature_eng_train_df.columns}")
            logging.info(f"Feature Engineering - Train Completed")
            
            # Transforming Data 
            numerical_cols,categorical_cols=self.separate_numerical_categorical_columns(df=feature_eng_train_df)
            
            # Saving column labels for prediction
            create_yaml_file_numerical_columns(column_list=numerical_cols,
                                               yaml_file_path=PREDICTION_YAML_FILE_PATH)
            
            create_yaml_file_categorical_columns_from_dataframe(dataframe=feature_eng_train_df,categorical_columns=categorical_cols,
                                                                yaml_file_path=PREDICTION_YAML_FILE_PATH)
            
            logging.info(f" Transformed Data Numerical Columns :{numerical_cols}")
            logging.info(f" Transformed Data Categorical Columns :{categorical_cols}")
            
            # peprocessing Categorical and NUmerical Columns
            data_preprocessor=DataProcessor(numerical_cols=numerical_cols,categorical_cols=categorical_cols)
            
            preprocessor=data_preprocessor.get_preprocessor()
        
            transformed_train_array=data_preprocessor.fit_transform(data=feature_eng_train_df)
            
            logging.info(f"Shape of the Transformed Data X_train: {transformed_train_array.shape} ")
            
            # Log the shape of Transformed Train
            logging.info("------- Transformed Data -----------")
            
            # Adding target column to transformed dataframe
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            
            
            os.makedirs(transformed_train_dir,exist_ok=True)
                                   
            ## Saving transformed train and test file
            logging.info("Saving Transformed Train and Transformed test Data")
            transformed_train_file_path = os.path.join(transformed_train_dir,"train.npy")

            save_numpy_array_data(file_path = transformed_train_file_path, array = transformed_train_array)


            logging.info("Transformation completed successfully")

                                ###############################################################
            
             ### Saving Feature engineering and preprocessor object 
            
            logging.info("Saving Feature Engineering Object")
            feature_engineering_object_file_path = self.data_transformation_config.feature_engineering_object_file_path
            save_object(file_path = feature_engineering_object_file_path,obj = fe_obj)
            save_object(file_path=os.path.join(ROOT_DIR,config_data[DATA_TRANSFORMATION_CONFIG_KEY][DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                                 os.path.basename(feature_engineering_object_file_path)),obj=fe_obj)


            ### Saving Feature engineering and preprocessor object 
            logging.info("Saving  Object")
            preprocessor_file_path = self.data_transformation_config.preprocessor_file_object_file_path
            save_object(file_path = preprocessor_file_path,obj = preprocessor)
            save_object(file_path=os.path.join(ROOT_DIR,config_data[DATA_TRANSFORMATION_CONFIG_KEY][DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                                 os.path.basename(preprocessor_file_path)),obj=preprocessor)

            data_transformation_artifact = DataTransformationArtifact(
                                                                        transformed_train_file_path =transformed_train_file_path,
                                                                        feature_engineering_object_file_path = feature_engineering_object_file_path)
            
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'*'*20} Data Transformation log completed {'*'*20}\n\n")