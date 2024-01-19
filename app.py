
from flask import Flask, render_template,redirect,request
from werkzeug.utils import secure_filename
from segmentation.utils.main_utils import read_yaml_file
from Prediction_code.batch import batch_prediction
from Prediction_code.instance import instance_prediction_class
import os
from segmentation.logger.logger import logging
from segmentation.constant import *
from segmentation.constant import *
#from Prediction_code.predict_dump import prediction_upload
from segmentation.pipeline.train import Pipeline
import shutil
import pandas as pd
import yaml

feature_engineering_file_path = os.path.join(os.getcwd(),'preprocessor','fea_eng.pkl')
transformer_file_path = os.path.join(os.getcwd(),'preprocessor','preprocessed.pkl')
model_file_path = "Saved_model/model.pkl"

BATCH_PREDICTION='batch_prediction'
UPLOAD_FOLDER = 'batch_prediction/Uploaded_CSV_FILE'

app = Flask(__name__, static_folder="static")
ALLOWED_EXTENSIONS = {'csv'}

SAVED_MODEL_DIRECTORY='Saved_model'


# Image file path
model_png_path = os.path.join(SAVED_MODEL_DIRECTORY,MODEL_NAME+'.png')
logging.info(f"Accessed Prediction image : {model_png_path}")

# Report path 
report_path=os.path.join(SAVED_MODEL_DIRECTORY, 'report.yaml')
report_data=read_yaml_file(report_path)

# csv file path 
csv_path=os.path.join(ROOT_DIR,SAVED_MODEL_DIRECTORY, 'data.csv')



@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/home")
def go_to_index():
    return redirect("index.html")

@app.route("/batch", methods=["GET","POST"])
def perform_batch_prediction():
    if request.method == 'GET':
        return render_template('batch.html')
    else:
        file = request.files['csv_file']  # Update the key to 'csv_file'
        directory = BATCH_PREDICTION

        if os.path.exists(directory):
            shutil.rmtree(directory, ignore_errors=True)
            print(f"Directory '{directory}' deleted.")
        else:
            print(f"Directory '{directory}' does not exist.")
        
        
        
        # Directory path
        directory_path = UPLOAD_FOLDER
        # Create the directory
        os.makedirs(directory_path, exist_ok=True)

        # Check if the file has a valid extension
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            # Delete all files in the file path
            for filename in os.listdir(os.path.join(UPLOAD_FOLDER)):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # Save the new file to the uploads directory
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            print(file_path)

            logging.info("CSV received and Uploaded")

            # Perform batch prediction using the uploaded file
            batch = batch_prediction(file_path, model_file_path, transformer_file_path, feature_engineering_file_path)
            prediction_csv_path=batch.start_batch_prediction()
            
            
            # ------------------------------------------------- Cluster Labelling -------------------------------------------#
            
            
            # Load YAML data from file
            with open('cluster_label.yaml', 'r') as file:
                yaml_data = yaml.safe_load(file)

            # Convert YAML data to dictionary
            data_dict = {int(key): value for key, value in yaml_data.items()}
            
            # Displaying disct data 
            logging.info(f" Data in cluster report : {data_dict}")

            # Create a DataFrame with the "prediction" column
            prediction_df = pd.read_csv(prediction_csv_path)

            # Assign the values from data_dict to the "prediction" column of your existing DataFrame
            prediction_df['Customer_cluster'] = prediction_df['cluster'].map(data_dict)
            
            # Saving csv after Mapping 
            prediction_df.to_csv(prediction_csv_path)
            
            
            
            
            output = "Batch Prediction Done "
            return render_template("batch.html", prediction_result=output,prediction_type='batch')
        else:
            return render_template('batch.html', prediction_type='batch', error='Invalid file type')

@app.route('/instance', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        prediction_data = read_yaml_file(PREDICTION_YAML_FILE_PATH)
        numerical_columns = prediction_data['numerical_columns']
        categorical_columns = prediction_data['categorical_columns']
        return render_template('instance.html', numerical_columns=numerical_columns, categorical_columns=categorical_columns, prediction_type='instance', predicted_expense='0')
    else:
        prediction_data = read_yaml_file(PREDICTION_YAML_FILE_PATH)
        numerical_columns = prediction_data['numerical_columns']
        categorical_columns = prediction_data['categorical_columns']

        numerical_inputs = {}
        categorical_inputs = {}
        for column in numerical_columns:
            numerical_inputs[column] = float(request.form[column])
        for category, values in categorical_columns.items():
            categorical_inputs[category] = request.form[category]

        logging.info("All data taken")

        df = pd.DataFrame(columns=numerical_columns + list(categorical_columns.keys()))
        # Create a dictionary with the data for the new row
        new_row = {**numerical_inputs, **categorical_inputs}

        # Initialize a DataFrame with the new row data
        df = pd.DataFrame([new_row])

        # Create an instance of your predictor class using the DataFrame
        predictor = instance_prediction_class(df=df)

        result = predictor.predict_expense()

        predicted_result = result

        logging.info("Prediction done")

        return render_template('instance.html', numerical_columns=numerical_columns, categorical_columns=categorical_columns,
                               prediction_type='instance', predicted_expense=predicted_result)



@app.route('/train_pipeline', methods=['GET', 'POST'])
def train_pipeline():
    try:
        pipeline=Pipeline()
        pipeline.run_pipeline()
        return render_template('index.html', message="Training complete")
    except Exception as e:
        logging.error(str(e))
        error_message = "An error occurred during training. Please try again."
        return render_template('index.html', error=error_message)


@app.route('/train', methods=['GET', 'POST'])
def train():
    try:
        
        num_clusters = report_data['parameters']['no_of_clusters']
        
        # Copying Prediction images from Saved_model directory 
        destination_image_path = './static/images/prediction.png'
        shutil.copy(model_png_path,destination_image_path)

        label_map = {}
        for i in range(num_clusters):
            label = request.form.get(f"label{i}")
            label_map[i] = label

        # Save the label mapping as a dictionary in YAML format
        yaml_filepath = "cluster_label.yaml"
        with open(yaml_filepath, 'w') as yaml_file:
            yaml.dump(label_map, yaml_file)

        return render_template('train.html', message="Training complete", num_clusters=num_clusters)

    except Exception as e:
        logging.error(str(e))
        error_message = "An error occurred during training. Please try again."
        return render_template('train.html', error=error_message)





if __name__ == '__main__':
    host = '0.0.0.0'  # Specify the host address you want to use
    port = 8050  # Specify the port number you want to use
    app.run(debug=True, host=host, port=port)