from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sklearn
import pickle

# Load models
crop_model = pickle.load(open('crop_model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get data from HTML form
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    
    # Create input data
    input_data = pd.DataFrame({
        'N': [N], 'P': [P], 'K': [K],
        'temperature': [temperature], 'humidity': [humidity],
        'ph': [ph], 'rainfall': [rainfall]
    })
    
    # Make prediction
    prediction = crop_model.predict(input_data)
    crop_name = le.inverse_transform(prediction)[0]
    
    result = "{} is the most suitable crop for this field".format(crop_name)
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True) 
 