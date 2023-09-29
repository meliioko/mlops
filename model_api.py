from flask import Flask, request
import json
import joblib
import numpy as np


app = Flask(__name__)

model = joblib.load('regression.joblib')

@app.route("/fake-predict")
def prediction():
    y_pred = [50000]
    return json.dumps({"prediction": y_pred})


@app.route("/predict", methods=['POST']) # permet de spécifier quand la fonction juste en dessous doit être appelée
def predict():

    # suppose que le client envoie une requête avec dedans X
    X = request.get_json()['X']
    X = np.array(X)
    pred = model.predict(X)
    return json.dumps({'prediction': pred[0]})

if __name__ == "__main__":
    app.run("0.0.0.0")