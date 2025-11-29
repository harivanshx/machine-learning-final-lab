"""
Preprocessing Module
Handles outlier detection and treatment
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class to handle data preprocessing operations"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataPreprocessor
        
        Args:
            df: Input DataFrame
        """
        self.df = df.copy()
        self.numeric_columns = None
        self.outlier_info = {}
        
    def identify_numeric_columns(self) -> List[str]:
        """
        Identify numeric columns in the dataset
        
        Returns:
            List of numeric column names
        """
        try:
            self.numeric_columns = self.df.select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist()
            
            # Remove ID columns
            self.numeric_columns = [
                col for col in self.numeric_columns 
                if 'ID' not in col.upper()
            ]
            
            logger.info(f"Identified {len(self.numeric_columns)} numeric columns")
            return self.numeric_columns
            
        except Exception as e:
            logger.error(f"Error identifying numeric columns: {e}")
            raise
    
    def detect_outliers_iqr(self, column: str, multiplier: float = 1.5) -> Tuple[np.ndarray, dict]:
        """
        Detect outliers using IQR method
        
        Args:
            column: Column name
            multiplier: IQR multiplier (default: 1.5)
            
        Returns:
            Tuple of (outlier_mask, outlier_info)
        """
        try:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outlier_mask = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
            
            outlier_info = {
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'n_outliers': outlier_mask.sum(),
                'percentage': (outlier_mask.sum() / len(self.df)) * 100
            }
            
            return outlier_mask, outlier_info
            
        except Exception as e:
            logger.error(f"Error detecting outliers in {column}: {e}")
            raise
    
    def treat_outliers(self, method: str = 'cap') -> pd.DataFrame:
        """
        Treat outliers in numeric columns
        
        Args:
            method: Treatment method ('cap', 'remove', or 'transform')
            
        Returns:
            DataFrame with treated outliers
        """
        try:
            if self.numeric_columns is None:
                self.identify_numeric_columns()
            
            logger.info(f"Treating outliers using method: {method}")
            
            for column in self.numeric_columns:
                outlier_mask, info = self.detect_outliers_iqr(column)
                self.outlier_info[column] = info
                
                if info['n_outliers'] > 0:
                    logger.info(
                        f"{column}: {info['n_outliers']} outliers "
                        f"({info['percentage']:.2f}%)"
                    )
                    
                    if method == 'cap':
                        # Cap outliers at bounds
                        self.df.loc[
                            self.df[column] < info['lower_bound'], column
                        ] = info['lower_bound']
                        self.df.loc[
                            self.df[column] > info['upper_bound'], column
                        ] = info['upper_bound']
                        
                    elif method == 'remove':
                        # Remove outlier rows
                        self.df = self.df[~outlier_mask]
                        
                    elif method == 'transform':
                        # Log transformation for positive values
                        if (self.df[column] > 0).all():
                            self.df[column] = np.log1p(self.df[column])
            
            logger.info(f"Outlier treatment completed. Final shape: {self.df.shape}")
            return self.df
            
        except Exception as e:
            logger.error(f"Error treating outliers: {e}")
            raise
    
    def get_outlier_report(self) -> pd.DataFrame:
        """
        Generate outlier detection report
        
        Returns:
            DataFrame containing outlier statistics
        """
        if not self.outlier_info:
            logger.warning("No outlier information available. Run treat_outliers() first.")
            return pd.DataFrame()
        
        report_data = []
        for column, info in self.outlier_info.items():
            report_data.append({
                'Column': column,
                'Q1': info['Q1'],
                'Q3': info['Q3'],
                'IQR': info['IQR'],
                'Lower_Bound': info['lower_bound'],
                'Upper_Bound': info['upper_bound'],
                'N_Outliers': info['n_outliers'],
                'Percentage': info['percentage']
            })
        
        return pd.DataFrame(report_data)


def main():
    """Main function for testing the preprocessor"""
    try:
        # Example usage
        from data_loader import DataLoader
        
        loader = DataLoader('Hotel Reservations.csv')
        df = loader.load_data()
        
        preprocessor = DataPreprocessor(df)
        preprocessor.identify_numeric_columns()
        df_processed = preprocessor.treat_outliers(method='cap')
        
        report = preprocessor.get_outlier_report()
        logger.info(f"Outlier report:\n{report}")
        
        return df_processed
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
