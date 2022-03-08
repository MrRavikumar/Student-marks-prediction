import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import math

app = Flask(__name__)
model = joblib.load(open('Student_marks_predictor_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods= ['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = np.array(input_features)
    output = model.predict([features_value])[0][0].round(2)
    return render_template('index.html', prediction_text= "You will get [{}%]  ".format(math.floor(output)))



if __name__ == "__main__":
    app.run(debug = True)
