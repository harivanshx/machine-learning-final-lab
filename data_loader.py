"""
Data Loader Module
Handles loading and initial validation of the dataset
"""

import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Class to handle data loading operations"""
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to the CSV file
        """
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            Exception: For other loading errors
        """
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            
            return self.df
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self) -> bool:
        """
        Validate the loaded data
        
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        try:
            if self.df is None:
                raise ValueError("No data loaded. Call load_data() first.")
            
            logger.info("Validating data...")
            
            # Check for empty dataframe
            if self.df.empty:
                raise ValueError("Loaded dataframe is empty")
            
            # Log basic information
            logger.info(f"Number of rows: {len(self.df)}")
            logger.info(f"Number of columns: {len(self.df.columns)}")
            logger.info(f"Columns: {list(self.df.columns)}")
            
            # Check for missing values
            missing_counts = self.df.isnull().sum()
            if missing_counts.sum() > 0:
                logger.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
            
            logger.info("Data validation completed successfully")
            return True
            
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            raise
    
    def get_data_info(self) -> dict:
        """
        Get summary information about the dataset
        
        Returns:
            Dictionary containing dataset information
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum()
        }
        
        return info


def main():
    """Main function for testing the data loader"""
    try:
        # Example usage
        loader = DataLoader('Hotel Reservations.csv')
        df = loader.load_data()
        loader.validate_data()
        
        info = loader.get_data_info()
        logger.info(f"Dataset info: {info}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
