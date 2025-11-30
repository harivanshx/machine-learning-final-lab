"""
Model Loader Module
Handles loading trained models and making predictions in production
"""

import pickle
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Union, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Production-ready model loader for inference"""
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize ModelLoader
        
        Args:
            model_dir: Directory containing saved models
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scaler = None
        self.feature_names = None
        
    def load_model(self, model_name: str) -> Any:
        """
        Load a specific model
        
        Args:
            model_name: Name of the model ('logistic_regression', 'random_forest', 'xgboost')
            
        Returns:
            Loaded model object
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        try:
            model_path = self.model_dir / f'{model_name}.pkl'
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            logger.info(f"Loading model from {model_path}")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self.models[model_name] = model
            logger.info(f"Successfully loaded {model_name}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def load_scaler(self) -> Any:
        """
        Load the feature scaler
        
        Returns:
            Loaded scaler object
        """
        try:
            scaler_path = self.model_dir / 'scaler.pkl'
            
            if not scaler_path.exists():
                logger.warning("Scaler not found. Some models may not work correctly.")
                return None
            
            logger.info(f"Loading scaler from {scaler_path}")
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            logger.info("Successfully loaded scaler")
            return self.scaler
            
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            raise
    
    def load_all_models(self) -> Dict[str, Any]:
        """
        Load all available models
        
        Returns:
            Dictionary of loaded models
        """
        try:
            model_names = ['logistic_regression', 'random_forest', 'xgboost']
            
            for model_name in model_names:
                try:
                    self.load_model(model_name)
                except FileNotFoundError:
                    logger.warning(f"Model {model_name} not found, skipping")
            
            self.load_scaler()
            
            logger.info(f"Loaded {len(self.models)} models")
            return self.models
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def prepare_features(self, data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """
        Prepare features for prediction
        
        Args:
            data: Input data (DataFrame or dictionary)
            
        Returns:
            Prepared feature DataFrame
        """
        try:
            # Convert dict to DataFrame if needed
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            # Ensure we have a copy
            df = data.copy()
            
            logger.info(f"Preparing features for {len(df)} samples")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def predict(self, 
                data: Union[pd.DataFrame, Dict], 
                model_name: str = 'random_forest',
                return_proba: bool = False) -> Union[np.ndarray, Dict]:
        """
        Make predictions using a loaded model
        
        Args:
            data: Input data for prediction
            model_name: Name of model to use
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions or probabilities
        """
        try:
            # Check if model is loaded
            if model_name not in self.models:
                logger.info(f"Model {model_name} not loaded, loading now...")
                self.load_model(model_name)
            
            model = self.models[model_name]
            
            # Prepare features
            X = self.prepare_features(data)
            
            # Scale features if using logistic regression
            if model_name == 'logistic_regression':
                if self.scaler is None:
                    self.load_scaler()
                X_scaled = self.scaler.transform(X)
                X_pred = X_scaled
            else:
                X_pred = X
            
            # Make predictions
            if return_proba:
                predictions = model.predict_proba(X_pred)
                logger.info(f"Generated probability predictions for {len(X)} samples")
            else:
                predictions = model.predict(X_pred)
                logger.info(f"Generated predictions for {len(X)} samples")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_with_confidence(self, 
                               data: Union[pd.DataFrame, Dict],
                               model_name: str = 'random_forest') -> List[Dict]:
        """
        Make predictions with confidence scores
        
        Args:
            data: Input data for prediction
            model_name: Name of model to use
            
        Returns:
            List of prediction dictionaries with confidence scores
        """
        try:
            # Get predictions and probabilities
            predictions = self.predict(data, model_name, return_proba=False)
            probabilities = self.predict(data, model_name, return_proba=True)
            
            results = []
            for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
                result = {
                    'prediction': int(pred),
                    'prediction_label': 'Canceled' if pred == 1 else 'Not Canceled',
                    'confidence': float(max(proba)),
                    'probability_not_canceled': float(proba[0]),
                    'probability_canceled': float(proba[1])
                }
                results.append(result)
            
            logger.info(f"Generated {len(results)} predictions with confidence scores")
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions with confidence: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'loaded_models': list(self.models.keys()),
            'scaler_loaded': self.scaler is not None,
            'model_directory': str(self.model_dir)
        }
        
        return info


def main():
    """Example usage of ModelLoader"""
    try:
        # Initialize loader
        loader = ModelLoader()
        
        # Load all models
        models = loader.load_all_models()
        
        # Example prediction data
        sample_data = {
            'no_of_adults': 2,
            'no_of_children': 0,
            'no_of_weekend_nights': 1,
            'no_of_week_nights': 2,
            'required_car_parking_space': 0,
            'lead_time': 224,
            'arrival_year': 2018,
            'arrival_month': 10,
            'arrival_date': 2,
            'repeated_guest': 0,
            'no_of_previous_cancellations': 0,
            'no_of_previous_bookings_not_canceled': 0,
            'avg_price_per_room': 65.0,
            'no_of_special_requests': 0,
            'type_of_meal_plan_label': 0,
            'room_type_reserved_label': 0,
            'market_segment_type_label': 3
        }
        
        # Make prediction
        result = loader.predict_with_confidence(sample_data, model_name='random_forest')
        
        logger.info(f"Prediction result: {result}")
        
        # Get model info
        info = loader.get_model_info()
        logger.info(f"Model info: {info}")
        
        return loader
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
