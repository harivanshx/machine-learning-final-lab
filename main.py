"""
Main Pipeline Script
Orchestrates the entire ML pipeline from data loading to model evaluation
"""

import logging
import argparse
from pathlib import Path
import sys

from data_loader import DataLoader
from preprocess import DataPreprocessor
from feature_engineering import FeatureEngineer
from train import ModelTrainer
from evaluate import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MLPipeline:
    """Main ML Pipeline orchestrator"""
    
    def __init__(self, config: dict):
        """
        Initialize MLPipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.df = None
        self.trainer = None
        self.evaluator = None
        
    def run(self):
        """Run the complete ML pipeline"""
        try:
            logger.info("="*80)
            logger.info("Starting ML Pipeline")
            logger.info("="*80)
            
            # Step 1: Load Data
            logger.info("\n" + "="*80)
            logger.info("STEP 1: Loading Data")
            logger.info("="*80)
            self.load_data()
            
            # Step 2: Preprocess Data
            logger.info("\n" + "="*80)
            logger.info("STEP 2: Preprocessing Data")
            logger.info("="*80)
            self.preprocess_data()
            
            # Step 3: Feature Engineering
            logger.info("\n" + "="*80)
            logger.info("STEP 3: Feature Engineering")
            logger.info("="*80)
            self.engineer_features()
            
            # Step 4: Train Models
            logger.info("\n" + "="*80)
            logger.info("STEP 4: Training Models")
            logger.info("="*80)
            self.train_models()
            
            # Step 5: Evaluate Models
            logger.info("\n" + "="*80)
            logger.info("STEP 5: Evaluating Models")
            logger.info("="*80)
            self.evaluate_models()
            
            logger.info("\n" + "="*80)
            logger.info("Pipeline completed successfully!")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def load_data(self):
        """Load and validate data"""
        try:
            loader = DataLoader(self.config['data_path'])
            self.df = loader.load_data()
            loader.validate_data()
            
            # Log data info
            info = loader.get_data_info()
            logger.info(f"Dataset shape: {info['shape']}")
            logger.info(f"Number of features: {len(info['columns'])}")
            
        except Exception as e:
            logger.error(f"Error in load_data: {e}")
            raise
    
    def preprocess_data(self):
        """Preprocess data (outlier treatment)"""
        try:
            preprocessor = DataPreprocessor(self.df)
            preprocessor.identify_numeric_columns()
            
            self.df = preprocessor.treat_outliers(
                method=self.config.get('outlier_method', 'cap')
            )
            
            # Log outlier report
            report = preprocessor.get_outlier_report()
            logger.info(f"Outlier treatment summary:\n{report.to_string(index=False)}")
            
            # Save preprocessed data if requested
            if self.config.get('save_intermediate', False):
                output_path = Path('intermediate_data')
                output_path.mkdir(exist_ok=True)
                self.df.to_csv(output_path / 'preprocessed_data.csv', index=False)
                logger.info("Saved preprocessed data")
            
        except Exception as e:
            logger.error(f"Error in preprocess_data: {e}")
            raise
    
    def engineer_features(self):
        """Engineer features and encode categorical variables"""
        try:
            engineer = FeatureEngineer(self.df)
            
            # Create new features
            self.df = engineer.create_features()
            logger.info(f"Created {len(engineer.new_features)} new features")
            
            # Encode categorical features
            self.df = engineer.encode_categorical_features(
                method=self.config.get('encoding_method', 'label')
            )
            
            # Log feature info
            feature_info = engineer.get_feature_list()
            logger.info(f"Total features: {feature_info['total_features']}")
            logger.info(f"Numeric features: {len(feature_info['numeric'])}")
            logger.info(f"Categorical features: {len(feature_info['categorical'])}")
            
            # Save engineered data if requested
            if self.config.get('save_intermediate', False):
                output_path = Path('intermediate_data')
                output_path.mkdir(exist_ok=True)
                self.df.to_csv(output_path / 'engineered_data.csv', index=False)
                logger.info("Saved engineered data")
            
        except Exception as e:
            logger.error(f"Error in engineer_features: {e}")
            raise
    
    def train_models(self):
        """Train all models"""
        try:
            self.trainer = ModelTrainer(
                self.df,
                target_col=self.config.get('target_col', 'booking_status_encoded')
            )
            
            # Prepare data
            self.trainer.prepare_data(
                test_size=self.config.get('test_size', 0.2),
                random_state=self.config.get('random_state', 42)
            )
            
            # Train models
            models = self.trainer.train_all_models()
            logger.info(f"Trained {len(models)} models")
            
            # Save models if requested
            if self.config.get('save_models', True):
                self.trainer.save_models(
                    output_dir=self.config.get('model_dir', 'models')
                )
            
        except Exception as e:
            logger.error(f"Error in train_models: {e}")
            raise
    
    def evaluate_models(self):
        """Evaluate all models"""
        try:
            self.evaluator = ModelEvaluator(
                self.trainer.models,
                self.trainer.X_test,
                self.trainer.y_test,
                self.trainer.scaler
            )
            
            # Evaluate all models
            results_df = self.evaluator.evaluate_all_models()
            
            # Generate visualizations
            self.evaluator.generate_confusion_matrices(
                output_dir=self.config.get('results_dir', 'results')
            )
            
            # Save results
            self.evaluator.save_results(
                output_dir=self.config.get('results_dir', 'results')
            )
            
            # Get best model
            best_model, best_score = self.evaluator.get_best_model(
                metric=self.config.get('best_model_metric', 'F1-Score')
            )
            
            logger.info(f"\nBest Model: {best_model}")
            logger.info(f"Best Score: {best_score:.4f}")
            
        except Exception as e:
            logger.error(f"Error in evaluate_models: {e}")
            raise


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run ML Pipeline for Hotel Reservations'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='Hotel Reservations.csv',
        help='Path to the data file'
    )
    
    parser.add_argument(
        '--outlier-method',
        type=str,
        choices=['cap', 'remove', 'transform'],
        default='cap',
        help='Method for outlier treatment'
    )
    
    parser.add_argument(
        '--encoding-method',
        type=str,
        choices=['label', 'onehot'],
        default='label',
        help='Method for categorical encoding'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of test set'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='Save intermediate data files'
    )
    
    parser.add_argument(
        '--save-models',
        action='store_true',
        default=True,
        help='Save trained models'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save models'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Create configuration
        config = {
            'data_path': args.data_path,
            'outlier_method': args.outlier_method,
            'encoding_method': args.encoding_method,
            'test_size': args.test_size,
            'random_state': args.random_state,
            'save_intermediate': args.save_intermediate,
            'save_models': args.save_models,
            'model_dir': args.model_dir,
            'results_dir': args.results_dir,
            'target_col': 'booking_status_encoded',
            'best_model_metric': 'F1-Score'
        }
        
        # Run pipeline
        pipeline = MLPipeline(config)
        pipeline.run()
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
