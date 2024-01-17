import sys
import os
import pandas as pd
from segmentation.logger.logger import logging
from segmentation.exception.exception import ApplicationException
from segmentation.utils.main_utils import save_object,read_yaml_file,load_numpy_array_data
from segmentation.entity.config_entity import ModelTrainerConfig
from segmentation.entity.artifact_entity import DataTransformationArtifact
from segmentation.entity.artifact_entity import ModelTrainerArtifact
from segmentation.constant import *
import sys
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import yaml
from sklearn.decomposition import PCA

    


class Model:
    def __init__(self,model_png_location):
    
        # png location 
        self.model_png_location=model_png_location
        file_location=self.model_png_location
        os.makedirs(file_location,exist_ok=True)
        
    
    
    def cluster_plot(self, df, file_path, cluster_column, model_name, x_col='Total_Amount', y_col='Income'):
        # Set the style of the plot
        sns.set(style="whitegrid")

        # Define the color palette for the clusters
        cluster_palette = sns.color_palette("Set2", df[cluster_column].nunique())

        # Create a figure with two subplots (1 row, 2 columns)
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        # Scatter plot
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=df[cluster_column], palette=cluster_palette, s=80, ax=axs[0])

        # Set the axis labels and title for the scatter plot
        axs[0].set_xlabel(x_col, fontsize=12)
        axs[0].set_ylabel(y_col, fontsize=12)
        axs[0].set_title("Cluster Plot", fontsize=14)

        # Display the legend for the scatter plot
        axs[0].legend(title=cluster_column)

        # Box plot
        sns.set(style="whitegrid")

        # Create the box plot using stripplot
        pal = sns.color_palette("Set2", df[cluster_column].nunique())
        pl = sns.stripplot(x=df[cluster_column], y=df[y_col], color="gray", alpha=0.5, jitter=True, size=4, ax=axs[1])
        pl = sns.boxenplot(x=df[cluster_column], y=df[y_col], palette=pal, ax=axs[1], legend=False)
        pl.set_title("Boxplot of customers clusters", pad=10, size=15)

        # Adjust the plot layout for the box plot
        axs[1].set_ylim(axs[0].get_ylim())  # Match y-axis limits with scatter plot
        axs[1].set_ylabel('')  # Remove y-axis label for better alignment

        # Save the figure
        filename = os.path.join(file_path, model_name + '.png')

        # Adjust the overall plot layout and save the figure
        fig.tight_layout()
        plt.savefig(filename)

        return filename
    

    def GaussianMixtureClustering(self, data):
        # Perform Gaussian Mixture Model clustering with the desired number of clusters
        model = GaussianMixture(n_components=CLUSTERS,init_params=INIT_PARAMS,covariance_type=COVARIANCE,
                                tol=TOLERANCE,max_iter=ITERATIONS)
        model.fit(data)
        labels = model.predict(data)

        logging.info("Labels created")

        # Return the cluster labels and the trained model
        return labels, model

    def adding_labels_to_data(self, data, cluster_labels):
        # Convert cluster_labels to a DataFrame
        logging.info("Converting cluster_labels to DataFrame")
        df_cluster_labels = pd.DataFrame(cluster_labels, columns=['cluster'])

        # Adding clutser column 
        df_gmm=data
        
        # Add cluster labels to the data
        df_gmm['cluster'] = df_cluster_labels['cluster']
        logging.info("Cluster added to the DataFrame")

        return df_gmm
    
    def calculate_silhouette_score(self, data, cluster_labels):
        # Calculate the silhouette score
        silhouette_avg = silhouette_score(data, cluster_labels)
        logging.info(f"Silhouette Score: {silhouette_avg}")
        return silhouette_avg
    
    
    
class ModelTrainer:

    def __init__(self, 
                 model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            
            # Accessing Artifacts
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

            
            ## Schema Yaml 
            self.schema_data=read_yaml_file(SCHEMA_FILE_PATH)
            
            
        except Exception as e:
            raise ApplicationException(e, sys) from e


    def save_model_and_report(self,best_model, model_report, model_file_path,report_file_path):
        # Saving Model Report and Model object
        save_object(file_path=model_file_path, obj=best_model)

        # Save the report as a YAML file
        with open(report_file_path, 'w') as file:
            yaml.dump(model_report, file)
        logging.info("Model and report saved to Artifact Folder.")
        

        # Save the report as a params.yaml
        file_path = os.path.join('params.yaml')
        with open(file_path, 'w') as file:
            yaml.dump(model_report, file)
        logging.info("Params.yaml file saved to the directory.")


        
    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info("Finding transformed Training data")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            
            logging.info("Transformed Data found!!!")
            train_array = load_numpy_array_data(file_path=transformed_train_file_path)
            
            file_path=self.data_transformation_artifact.feature_eng_df
            logging.info("Feature Engineering Csv Loading ..")
            train_df=pd.read_csv(file_path)
            
            logging.info( f" Shape of the  trained data : {train_array.shape}")
            
            
            # PCA transformation 
            logging.info(" PCA Transformation Started ........")
            # Setting n_components=3
            pca = PCA(n_components=3)  # Create an instance of the PCA class
            pca.fit(train_array)  # Fit the PCA model to the data
            transformed_data = pca.transform(train_array) 
            logging.info(" PCA Transformation Completed ........")
            
            
            
            # Model_trainer
            logging.info(" Training Kmeans.....")
            model=Model(model_png_location=self.model_trainer_config.png_location)
              
            # GMM 
            cluster_labels,gmm_model=model.GaussianMixtureClustering(data=transformed_data)
            
            score=model.calculate_silhouette_score(data=transformed_data,cluster_labels=cluster_labels)
            
            df_gmm=model.adding_labels_to_data(data=train_df,cluster_labels=cluster_labels)
        
            prediction_png_gmm_path=model.cluster_plot(df=df_gmm,file_path=self.model_trainer_config.png_location,
                                                  model_name=MODEL_NAME,cluster_column='cluster')
                                                                                                     

            
    

            logging.info(f"Saving metrics of model  : MODEL_NAME")
            model_report={
                "Model_name":MODEL_NAME,
                METRIC:str(score),
                "parameters":parameters
            }
            
            logging.info("Report Created")
            
            logging.info(f"Dumping Metrics in report.....")
            
             # Save report in artifact folder
            report_file_path = self.model_trainer_config.model_report_path
            save_model_path=self.model_trainer_config.trained_model_file_path
            png_path=self.model_trainer_config.png_location
            
            
            # Saving Model Object and Model Report 
            self.save_model_and_report(best_model=gmm_model,
                                        model_report=model_report,
                                        model_file_path=save_model_path,
                                        report_file_path=report_file_path)
            
            logging.info("Model and Report Saved")

            
            model_trainer_artifact = ModelTrainerArtifact(
                                                model_selected=save_model_path,
                                                report_path=report_file_path,
                                                model_prediction_png=png_path
                                            )
            

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")
            
            