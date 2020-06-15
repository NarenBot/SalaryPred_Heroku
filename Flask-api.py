# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 12:30:35 2020

@author: Naren
"""

import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

pickle_in = open('Model.pkl', 'rb')
model = pickle.load(pickle_in)
#html_link = r"C:\Users\Naren\Anaconda3\Scripts(Spyder)\Salary Prediction\templates\index.html"
@app.route('/', methods=['Get'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_features = [ int(x) for x in request.form.values()]
    final_features = [ np.array(input_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0],2)
    return render_template('index.html', prediction_text = 'Employee Salary should be ${}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
    
