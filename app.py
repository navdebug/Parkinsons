# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:03:12 2021

@author: Ananya M Menon
"""


from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('parkinson.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    if output == 0:
        return render_template('home.html', prediction_text= 'Yayy! You won't suffer from parkinson's disease :)')
    else:
        return render_template('home.html', prediction_text= 'Opps. Walk more. You might suffer from Parkinson's disease :(')


if __name__ == "__main__":
    app.run(debug=True) 
