"""
Prediction API Example
Simple Flask API for serving model predictions
"""

from flask import Flask, request, jsonify
from model_loader import ModelLoader
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize model loader
model_loader = ModelLoader()
model_loader.load_all_models()

logger.info("Models loaded successfully")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': model_loader.get_model_info()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Expected JSON format:
    {
        "data": {
            "no_of_adults": 2,
            "no_of_children": 0,
            ...
        },
        "model": "random_forest"  # optional, defaults to random_forest
    }
    """
    try:
        # Get request data
        request_data = request.get_json()
        
        if 'data' not in request_data:
            return jsonify({
                'error': 'Missing required field: data'
            }), 400
        
        # Get model name (default to random_forest)
        model_name = request_data.get('model', 'random_forest')
        
        # Make prediction
        result = model_loader.predict_with_confidence(
            request_data['data'],
            model_name=model_name
        )
        
        return jsonify({
            'success': True,
            'prediction': result[0],
            'model_used': model_name
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    
    Expected JSON format:
    {
        "data": [
            {"no_of_adults": 2, ...},
            {"no_of_adults": 1, ...}
        ],
        "model": "random_forest"  # optional
    }
    """
    try:
        # Get request data
        request_data = request.get_json()
        
        if 'data' not in request_data:
            return jsonify({
                'error': 'Missing required field: data'
            }), 400
        
        # Get model name
        model_name = request_data.get('model', 'random_forest')
        
        # Make predictions
        results = model_loader.predict_with_confidence(
            request_data['data'],
            model_name=model_name
        )
        
        return jsonify({
            'success': True,
            'predictions': results,
            'count': len(results),
            'model_used': model_name
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify(model_loader.get_model_info())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
