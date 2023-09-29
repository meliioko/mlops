from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('tumor_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    size = data['size']
    p53_concentration = data['p53_concentration']

    X = scaler.transform([[size, p53_concentration]])
    prediction = model.predict(X)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
