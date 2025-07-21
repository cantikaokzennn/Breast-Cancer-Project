from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flasgger import Swagger, swag_from
import numpy as np
import joblib
import logging
from datetime import datetime

# Load model
model = joblib.load('MLP_model_final.pkl')

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Init Flask app
app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@swag_from({
    'tags': ['Prediction'],
    'summary': 'Predict Breast Cancer Classification',
    'description': 'Menerima input fitur dan mengembalikan hasil klasifikasi kanker payudara.',
    'consumes': ['application/json'],
    'produces': ['application/json'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'radius_mean': {'type': 'number', 'example': 14.2},
                    'perimeter_mean': {'type': 'number', 'example': 90.2},
                    'area_mean': {'type': 'number', 'example': 600.1},
                    'concavity_mean': {'type': 'number', 'example': 0.1},
                    'concave_points_mean': {'type': 'number', 'example': 0.05},
                    'area_se': {'type': 'number', 'example': 40.1},
                    'radius_worst': {'type': 'number', 'example': 16.4},
                    'perimeter_worst': {'type': 'number', 'example': 110.2},
                    'area_worst': {'type': 'number', 'example': 900.5},
                    'concave_points_worst': {'type': 'number', 'example': 0.15},
                },
                'required': [
                    'radius_mean', 'perimeter_mean', 'area_mean',
                    'concavity_mean', 'concave_points_mean', 'area_se',
                    'radius_worst', 'perimeter_worst', 'area_worst', 'concave_points_worst'
                ]
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Prediction result',
            'schema': {
                'type': 'object',
                'properties': {
                    'success': {'type': 'boolean', 'example': True},
                    'prediction': {'type': 'integer', 'example': 1},
                    'probability': {
                        'type': 'object',
                        'properties': {
                            'positive': {'type': 'number', 'example': 0.9123},
                            'negative': {'type': 'number', 'example': 0.0877}
                        }
                    },
                    'timestamp': {'type': 'string', 'example': '2025-06-15T12:34:56.789123'}
                }
            }
        },
        500: {
            'description': 'Internal server error',
            'schema': {
                'type': 'object',
                'properties': {
                    'success': {'type': 'boolean', 'example': False},
                    'error': {'type': 'string', 'example': 'Internal server error'}
                }
            }
        }
    }
})
def predict():
    try:
        data = request.json
        logging.info(f"Received prediction request: {data}")

        # Ambil nilai fitur dari body
        values = np.array([[ 
            float(data['radius_mean']),
            float(data['perimeter_mean']),
            float(data['area_mean']),
            float(data['concavity_mean']),
            float(data['concave_points_mean']),
            float(data['area_se']),
            float(data['radius_worst']),
            float(data['perimeter_worst']),
            float(data['area_worst']),
            float(data['concave_points_worst'])
        ]])

        # Prediksi
        predicted = model.predict(values)[0]
        probabilities = model.predict_proba(values)[0]

        label_classes = ['Jinak (B)', 'Ganas (M)']

        return jsonify({
            "success": True,
            'predicted_label': label_classes[predicted],
            'probabilities': {
                label_classes[0]: round(float(probabilities[0]), 4),
                label_classes[1]: round(float(probabilities[1]), 4)
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
