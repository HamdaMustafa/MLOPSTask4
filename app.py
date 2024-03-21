from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model_filename = 'gender_classification_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Assuming your model expects a numpy array of features
    features = np.array(data['features'])
    prediction = model.predict(features.reshape(1, -1))
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
