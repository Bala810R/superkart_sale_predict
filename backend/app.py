
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app
superkart_api = Flask("SuperKart Sales Predictor")

# Load the trained model
model = joblib.load("backend/superkart_sales_model.joblib")

# Define a route for the home page
@superkart_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Predictor"

# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the request
    dataset = request.get_json()

    # Extract relevant features from the input data
    sample = {
    'Product_Weight': dataset['Product_Weight'],
    'Product_Sugar_Content': dataset['Product_Sugar_Content'],
    'Product_Allocated_Area': dataset['Product_Allocated_Area'],
    'Product_MRP': dataset['Product_MRP'],
    'Store_Size': dataset['Store_Size'],
    'Store_Location_City_Type': dataset['Store_Location_City_Type'],
    'Store_Type': dataset['Store_Type'],
    'Product_Id_char': dataset['Product_Id_char'],
    'Store_Age_Years': dataset['Store_Age_Years'],
    'Product_Type_Category': dataset['Product_Type_Category']
}

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Make a prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]

    # Return the prediction as a JSON response
    return jsonify({'Sales': prediction})

# Define an endpoint to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the file into a DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for the batch data
    predictions = model.predict(input_data).tolist()

    # Create an output dictionary mapping row index to predicted sales
    output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

    return output_dict


# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
