import pickle
import numpy as np
from flask import Flask, request, jsonify
from perceptron import Perceptron

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict_get', methods=['GET'])
def get_prediction():
    try:
        sepal_length = float(request.args.get('sl'))
        petal_length = float(request.args.get('pl'))

        features = np.array([sepal_length, petal_length]).reshape(1, -1)

        predicted_class = int(model.predict(features))
        return jsonify(features=features.tolist(), predicted_class=predicted_class)
    except Exception as e:
        return jsonify(error=str(e)), 400

@app.route('/predict_post', methods=['POST'])
def post_predict():
    try:
        data = request.get_json(force=True)
        sepal_length = float(data.get('sl'))
        petal_length = float(data.get('pl'))

        features = np.array([sepal_length, petal_length]).reshape(1, -1)

        predicted_class = int(model.predict(features))
        output = dict(features=features.tolist(), predicted_class=predicted_class)
        return jsonify(output)
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)