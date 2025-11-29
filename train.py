"""
Training Module
Handles model training and hyperparameter tuning
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Tuple, Dict, Any
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class to handle model training operations"""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'booking_status_encoded'):
        """
        Initialize ModelTrainer
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
        """
        self.df = df.copy()
        self.target_col = target_col
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare data for training
        
        Args:
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            logger.info("Preparing data for training...")
            
            # Identify feature columns
            exclude_cols = [
                self.target_col,
                'booking_status',
                'Booking_ID'
            ]
            
            # Get numeric columns only
            numeric_cols = self.df.select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist()
            
            feature_cols = [
                col for col in numeric_cols 
                if col not in exclude_cols and 'ID' not in col.upper()
            ]
            
            logger.info(f"Using {len(feature_cols)} features")
            
            # Prepare X and y
            X = self.df[feature_cols]
            y = self.df[self.target_col]
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )
            
            logger.info(f"Training set: {self.X_train.shape}")
            logger.info(f"Test set: {self.X_test.shape}")
            logger.info(f"Class distribution in training set:\n{self.y_train.value_counts()}")
            
            return self.X_train, self.X_test, self.y_train, self.y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def scale_features(self) -> Tuple:
        """
        Scale features using StandardScaler
        
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        try:
            logger.info("Scaling features...")
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            X_test_scaled = self.scaler.transform(self.X_test)
            
            logger.info("Feature scaling completed")
            
            return X_train_scaled, X_test_scaled
            
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            raise
    
    def train_logistic_regression(self, use_scaled: bool = True, **kwargs) -> Any:
        """
        Train Logistic Regression model
        
        Args:
            use_scaled: Whether to use scaled features
            **kwargs: Additional parameters for LogisticRegression
            
        Returns:
            Trained model
        """
        try:
            logger.info("Training Logistic Regression...")
            
            # Default parameters
            params = {
                'max_iter': 1000,
                'random_state': 42,
                'class_weight': 'balanced'
            }
            params.update(kwargs)
            
            model = LogisticRegression(**params)
            
            if use_scaled:
                X_train_scaled, _ = self.scale_features()
                model.fit(X_train_scaled, self.y_train)
            else:
                model.fit(self.X_train, self.y_train)
            
            self.models['Logistic Regression'] = model
            logger.info("Logistic Regression training completed")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training Logistic Regression: {e}")
            raise
    
    def train_random_forest(self, **kwargs) -> Any:
        """
        Train Random Forest model
        
        Args:
            **kwargs: Additional parameters for RandomForestClassifier
            
        Returns:
            Trained model
        """
        try:
            logger.info("Training Random Forest...")
            
            # Default parameters
            params = {
                'n_estimators': 100,
                'random_state': 42,
                'class_weight': 'balanced',
                'n_jobs': -1
            }
            params.update(kwargs)
            
            model = RandomForestClassifier(**params)
            model.fit(self.X_train, self.y_train)
            
            self.models['Random Forest'] = model
            logger.info("Random Forest training completed")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
            raise
    
    def train_xgboost(self, **kwargs) -> Any:
        """
        Train XGBoost model
        
        Args:
            **kwargs: Additional parameters for XGBClassifier
            
        Returns:
            Trained model
        """
        try:
            logger.info("Training XGBoost...")
            
            # Calculate scale_pos_weight
            scale_pos_weight = (
                (self.y_train == 0).sum() / (self.y_train == 1).sum()
            )
            
            # Default parameters
            params = {
                'n_estimators': 100,
                'random_state': 42,
                'scale_pos_weight': scale_pos_weight,
                'n_jobs': -1
            }
            params.update(kwargs)
            
            model = XGBClassifier(**params)
            model.fit(self.X_train, self.y_train)
            
            self.models['XGBoost'] = model
            logger.info("XGBoost training completed")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")
            raise
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Train all models
        
        Returns:
            Dictionary of trained models
        """
        try:
            logger.info("Training all models...")
            
            self.train_logistic_regression()
            self.train_random_forest()
            self.train_xgboost()
            
            logger.info(f"Trained {len(self.models)} models")
            
            return self.models
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    def save_models(self, output_dir: str = 'models') -> None:
        """
        Save trained models to disk
        
        Args:
            output_dir: Directory to save models
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            logger.info(f"Saving models to {output_path}")
            
            for name, model in self.models.items():
                filename = output_path / f"{name.replace(' ', '_').lower()}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {name} to {filename}")
            
            # Save scaler if it exists
            if self.scaler is not None:
                scaler_file = output_path / 'scaler.pkl'
                with open(scaler_file, 'wb') as f:
                    pickle.dump(self.scaler, f)
                logger.info(f"Saved scaler to {scaler_file}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise


def main():
    """Main function for testing the trainer"""
    try:
        # Example usage
        from data_loader import DataLoader
        from preprocess import DataPreprocessor
        from feature_engineering import FeatureEngineer
        
        loader = DataLoader('Hotel Reservations.csv')
        df = loader.load_data()
        
        preprocessor = DataPreprocessor(df)
        df_processed = preprocessor.treat_outliers(method='cap')
        
        engineer = FeatureEngineer(df_processed)
        df_engineered = engineer.create_features()
        df_final = engineer.encode_categorical_features(method='label')
        
        trainer = ModelTrainer(df_final)
        trainer.prepare_data()
        models = trainer.train_all_models()
        trainer.save_models()
        
        logger.info("Training pipeline completed successfully")
        
        return trainer
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
