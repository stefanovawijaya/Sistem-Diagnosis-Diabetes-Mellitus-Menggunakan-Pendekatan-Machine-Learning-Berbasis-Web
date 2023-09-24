from flask import Flask, render_template, request, redirect
from sklearn.preprocessing import StandardScaler
import pickle
import sklearn
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':

        with open('model-diabetes.pkl', 'rb') as r:
            model = pickle.load(r)

        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bPressure = float(request.form['b-pressure'])
        sThickness = float(request.form['s-thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])

        data = np.array((pregnancies, glucose, bPressure, sThickness, insulin, bmi, dpf, age))
        data = np.reshape(data, (1, -1))

        dataset = pd.read_csv('diabetes.csv')
        X = dataset.drop(columns = 'Outcome', axis = 1) 
        scaler = StandardScaler()
        X = X.values
        scaler.fit(X)
        std_data = scaler.transform(data)

        hasDiabetes = model.predict(std_data)

        return render_template('result.html', finalData = hasDiabetes[0])
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)