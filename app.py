from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load('rf_acc_68.pkl')
scaler = joblib.load('normalizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form and convert to float
        values = [float(x) for x in request.form.values()]
        values = np.array(values).reshape(1, -1)

        # Normalize input
        values_scaled = scaler.transform(values)

        # Predict
        prediction = model.predict(values_scaled)[0]
        result = "Likely to have Liver Cirrhosis" if prediction == 1 else "Not Likely to have Liver Cirrhosis"
        return render_template('index.html', result=result)
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
