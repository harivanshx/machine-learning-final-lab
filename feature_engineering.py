"""
Feature Engineering Module
Handles feature creation and encoding
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class to handle feature engineering operations"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize FeatureEngineer
        
        Args:
            df: Input DataFrame
        """
        self.df = df.copy()
        self.label_encoders = {}
        self.new_features = []
        
    def create_features(self) -> pd.DataFrame:
        """
        Create new features from existing ones
        
        Returns:
            DataFrame with new features
        """
        try:
            logger.info("Creating new features...")
            
            # Total nights
            if 'no_of_weekend_nights' in self.df.columns and 'no_of_week_nights' in self.df.columns:
                self.df['total_nights'] = (
                    self.df['no_of_weekend_nights'] + self.df['no_of_week_nights']
                )
                self.new_features.append('total_nights')
                logger.info("Created feature: total_nights")
            
            # Total guests
            if 'no_of_adults' in self.df.columns and 'no_of_children' in self.df.columns:
                self.df['total_guests'] = (
                    self.df['no_of_adults'] + self.df['no_of_children']
                )
                self.new_features.append('total_guests')
                logger.info("Created feature: total_guests")
            
            # Price per night
            if 'avg_price_per_room' in self.df.columns and 'total_nights' in self.df.columns:
                self.df['total_price'] = (
                    self.df['avg_price_per_room'] * self.df['total_nights']
                )
                self.new_features.append('total_price')
                logger.info("Created feature: total_price")
            
            # Lead time category
            if 'lead_time' in self.df.columns:
                self.df['lead_time_category'] = pd.cut(
                    self.df['lead_time'],
                    bins=[-1, 7, 30, 90, float('inf')],
                    labels=['last_minute', 'short', 'medium', 'long']
                )
                self.new_features.append('lead_time_category')
                logger.info("Created feature: lead_time_category")
            
            # Is repeated guest
            if 'repeated_guest' in self.df.columns:
                self.df['is_repeated_guest'] = (self.df['repeated_guest'] == 1).astype(int)
                self.new_features.append('is_repeated_guest')
                logger.info("Created feature: is_repeated_guest")
            
            # Has special requests
            if 'no_of_special_requests' in self.df.columns:
                self.df['has_special_requests'] = (
                    self.df['no_of_special_requests'] > 0
                ).astype(int)
                self.new_features.append('has_special_requests')
                logger.info("Created feature: has_special_requests")
            
            # Booking history score
            if 'no_of_previous_bookings_not_canceled' in self.df.columns and \
               'no_of_previous_cancellations' in self.df.columns:
                total_bookings = (
                    self.df['no_of_previous_bookings_not_canceled'] + 
                    self.df['no_of_previous_cancellations']
                )
                self.df['booking_history_score'] = np.where(
                    total_bookings > 0,
                    self.df['no_of_previous_bookings_not_canceled'] / total_bookings,
                    0
                )
                self.new_features.append('booking_history_score')
                logger.info("Created feature: booking_history_score")
            
            # Weekend ratio
            if 'no_of_weekend_nights' in self.df.columns and 'total_nights' in self.df.columns:
                self.df['weekend_ratio'] = np.where(
                    self.df['total_nights'] > 0,
                    self.df['no_of_weekend_nights'] / self.df['total_nights'],
                    0
                )
                self.new_features.append('weekend_ratio')
                logger.info("Created feature: weekend_ratio")
            
            # Price per guest
            if 'avg_price_per_room' in self.df.columns and 'total_guests' in self.df.columns:
                self.df['price_per_guest'] = np.where(
                    self.df['total_guests'] > 0,
                    self.df['avg_price_per_room'] / self.df['total_guests'],
                    self.df['avg_price_per_room']
                )
                self.new_features.append('price_per_guest')
                logger.info("Created feature: price_per_guest")
            
            # Arrival month season
            if 'arrival_month' in self.df.columns:
                season_map = {
                    12: 'winter', 1: 'winter', 2: 'winter',
                    3: 'spring', 4: 'spring', 5: 'spring',
                    6: 'summer', 7: 'summer', 8: 'summer',
                    9: 'fall', 10: 'fall', 11: 'fall'
                }
                self.df['arrival_season'] = self.df['arrival_month'].map(season_map)
                self.new_features.append('arrival_season')
                logger.info("Created feature: arrival_season")
            
            logger.info(f"Created {len(self.new_features)} new features")
            return self.df
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            raise
    
    def encode_categorical_features(self, method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            method: Encoding method ('label' or 'onehot')
            
        Returns:
            DataFrame with encoded features
        """
        try:
            logger.info(f"Encoding categorical features using {method} encoding")
            
            # Identify categorical columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            
            # Remove target column if present
            if 'booking_status' in categorical_cols:
                categorical_cols.remove('booking_status')
            
            # Remove ID columns
            categorical_cols = [
                col for col in categorical_cols 
                if 'ID' not in col.upper()
            ]
            
            if method == 'label':
                for col in categorical_cols:
                    le = LabelEncoder()
                    self.df[f'{col}_label'] = le.fit_transform(self.df[col])
                    self.label_encoders[col] = le
                    logger.info(f"Label encoded: {col}")
            
            elif method == 'onehot':
                self.df = pd.get_dummies(
                    self.df,
                    columns=categorical_cols,
                    prefix=categorical_cols,
                    drop_first=True
                )
                logger.info(f"One-hot encoded {len(categorical_cols)} columns")
            
            # Encode target variable if present
            if 'booking_status' in self.df.columns:
                le = LabelEncoder()
                self.df['booking_status_encoded'] = le.fit_transform(
                    self.df['booking_status']
                )
                self.label_encoders['booking_status'] = le
                logger.info("Encoded target variable: booking_status")
            
            logger.info(f"Encoding completed. Final shape: {self.df.shape}")
            return self.df
            
        except Exception as e:
            logger.error(f"Error encoding features: {e}")
            raise
    
    def get_feature_list(self) -> Dict[str, List[str]]:
        """
        Get list of features by type
        
        Returns:
            Dictionary containing feature lists
        """
        numeric_features = self.df.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        
        categorical_features = self.df.select_dtypes(
            include=['object']
        ).columns.tolist()
        
        return {
            'numeric': numeric_features,
            'categorical': categorical_features,
            'new_features': self.new_features,
            'total_features': len(self.df.columns)
        }


def main():
    """Main function for testing the feature engineer"""
    try:
        # Example usage
        from data_loader import DataLoader
        from preprocess import DataPreprocessor
        
        loader = DataLoader('Hotel Reservations.csv')
        df = loader.load_data()
        
        preprocessor = DataPreprocessor(df)
        df_processed = preprocessor.treat_outliers(method='cap')
        
        engineer = FeatureEngineer(df_processed)
        df_engineered = engineer.create_features()
        df_final = engineer.encode_categorical_features(method='label')
        
        feature_info = engineer.get_feature_list()
        logger.info(f"Feature info: {feature_info}")
        
        return df_final
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
