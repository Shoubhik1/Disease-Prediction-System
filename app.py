
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
# Load trained model files
heart_model = pickle.load(open('models/heart_model.pkl', 'rb'))
diabetes_model = pickle.load(open('models/diabetes_model.pkl', 'rb'))
parkinsons_model = pickle.load(open('models/parkinsons_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/heart', methods=['GET', 'POST'])
def heart():
    prediction = None
    if request.method == 'POST':
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]
        input_data = np.array([features])
        result = heart_model.predict(input_data)
        prediction = "Person has Heart Disease" if result[0] == 1 else "Person does NOT have Heart Disease"
    return render_template('heart.html', prediction=prediction)


@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    prediction = None
    if request.method == 'POST':
        features = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        input_data = np.array([features])
        result = diabetes_model.predict(input_data)
        prediction = "Person has Diabetes" if result[0] == 1 else "Person does NOT have Diabetes"
    return render_template('diabetes.html', prediction=prediction)


@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    prediction = None
    if request.method == 'POST':
        features = [
            float(request.form['fo']),
            float(request.form['fhi']),
            float(request.form['flo']),
            float(request.form['Jitter_percent']),
            float(request.form['Jitter_Abs']),
            float(request.form['RAP']),
            float(request.form['PPQ']),
            float(request.form['DDP']),
            float(request.form['Shimmer']),
            float(request.form['Shimmer_dB']),
            float(request.form['APQ3']),
            float(request.form['APQ5']),
            float(request.form['APQ']),
            float(request.form['DDA']),
            float(request.form['NHR']),
            float(request.form['HNR']),
            float(request.form['RPDE']),
            float(request.form['DFA']),
            float(request.form['spread1']),
            float(request.form['spread2']),
            float(request.form['D2']),
            float(request.form['PPE'])
        ]
        input_data = np.array([features])
        result = parkinsons_model.predict(input_data)
        prediction = "Person has Parkinson's Disease" if result[0] == 1 else "Person does NOT have Parkinson's Disease"
    return render_template('parkinsons.html', prediction=prediction)


# New route for About page
@app.route('/about')
def about():
    return render_template('about.html')


# New route for Contact Us page
@app.route('/contact')
def contact():
    return render_template('contactus.html')


if __name__ == '__main__':
    app.run(debug=True)

