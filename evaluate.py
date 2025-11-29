"""
Evaluation Module
Handles model evaluation and performance metrics
"""

import pandas as pd
import numpy as np
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class to handle model evaluation operations"""
    
    def __init__(self, models: Dict[str, Any], X_test, y_test, scaler=None):
        """
        Initialize ModelEvaluator
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            scaler: Fitted scaler (optional)
        """
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.scaler = scaler
        self.results = {}
        self.predictions = {}
        
    def evaluate_model(self, model_name: str, model: Any, use_scaled: bool = False) -> Dict[str, float]:
        """
        Evaluate a single model
        
        Args:
            model_name: Name of the model
            model: Trained model
            use_scaled: Whether to use scaled features
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            logger.info(f"Evaluating {model_name}...")
            
            # Prepare test data
            if use_scaled and self.scaler is not None:
                X_test_eval = self.scaler.transform(self.X_test)
            else:
                X_test_eval = self.X_test
            
            # Make predictions
            y_pred = model.predict(X_test_eval)
            y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
            
            # Store predictions
            self.predictions[model_name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Calculate metrics
            metrics = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1-Score': f1_score(self.y_test, y_pred),
                'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba)
            }
            
            self.results[model_name] = metrics
            
            logger.info(f"{model_name} - Accuracy: {metrics['Accuracy']:.4f}")
            logger.info(f"{model_name} - F1-Score: {metrics['F1-Score']:.4f}")
            logger.info(f"{model_name} - ROC-AUC: {metrics['ROC-AUC']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            raise
    
    def evaluate_all_models(self) -> pd.DataFrame:
        """
        Evaluate all models
        
        Returns:
            DataFrame containing all evaluation metrics
        """
        try:
            logger.info("Evaluating all models...")
            
            for model_name, model in self.models.items():
                use_scaled = (model_name == 'Logistic Regression')
                self.evaluate_model(model_name, model, use_scaled=use_scaled)
            
            # Create results dataframe
            results_df = pd.DataFrame(self.results).T
            results_df.index.name = 'Model'
            results_df = results_df.reset_index()
            
            logger.info("\nModel Performance Comparison:")
            logger.info(f"\n{results_df.to_string(index=False)}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
            raise
    
    def generate_confusion_matrices(self, output_dir: str = 'results') -> None:
        """
        Generate and save confusion matrices for all models
        
        Args:
            output_dir: Directory to save plots
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            logger.info("Generating confusion matrices...")
            
            n_models = len(self.predictions)
            fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
            
            if n_models == 1:
                axes = [axes]
            
            for idx, (model_name, preds) in enumerate(self.predictions.items()):
                cm = confusion_matrix(self.y_test, preds['y_pred'])
                
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    ax=axes[idx],
                    cbar=False
                )
                axes[idx].set_title(f'{model_name}\nConfusion Matrix')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
            
            plt.tight_layout()
            
            filename = output_path / 'confusion_matrices.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrices to {filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating confusion matrices: {e}")
            raise
    
    def generate_classification_reports(self) -> Dict[str, str]:
        """
        Generate classification reports for all models
        
        Returns:
            Dictionary of classification reports
        """
        try:
            logger.info("Generating classification reports...")
            
            reports = {}
            for model_name, preds in self.predictions.items():
                report = classification_report(
                    self.y_test,
                    preds['y_pred'],
                    target_names=['Not Canceled', 'Canceled']
                )
                reports[model_name] = report
                
                logger.info(f"\n{model_name} Classification Report:\n{report}")
            
            return reports
            
        except Exception as e:
            logger.error(f"Error generating classification reports: {e}")
            raise
    
    def save_results(self, output_dir: str = 'results') -> None:
        """
        Save evaluation results to CSV
        
        Args:
            output_dir: Directory to save results
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            logger.info("Saving evaluation results...")
            
            # Save metrics
            results_df = pd.DataFrame(self.results).T
            results_df.index.name = 'Model'
            
            filename = output_path / 'model_comparison_results.csv'
            results_df.to_csv(filename)
            logger.info(f"Saved results to {filename}")
            
            # Save classification reports
            for model_name, report in self.generate_classification_reports().items():
                report_file = output_path / f'{model_name.replace(" ", "_").lower()}_report.txt'
                with open(report_file, 'w') as f:
                    f.write(report)
                logger.info(f"Saved {model_name} report to {report_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def get_best_model(self, metric: str = 'F1-Score') -> Tuple[str, float]:
        """
        Get the best performing model based on a metric
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, metric_value)
        """
        try:
            if not self.results:
                raise ValueError("No evaluation results available. Run evaluate_all_models() first.")
            
            best_model = max(
                self.results.items(),
                key=lambda x: x[1][metric]
            )
            
            logger.info(f"Best model by {metric}: {best_model[0]} ({best_model[1][metric]:.4f})")
            
            return best_model[0], best_model[1][metric]
            
        except Exception as e:
            logger.error(f"Error getting best model: {e}")
            raise


def main():
    """Main function for testing the evaluator"""
    try:
        # Example usage
        from data_loader import DataLoader
        from preprocess import DataPreprocessor
        from feature_engineering import FeatureEngineer
        from train import ModelTrainer
        
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
        
        evaluator = ModelEvaluator(
            models,
            trainer.X_test,
            trainer.y_test,
            trainer.scaler
        )
        
        results_df = evaluator.evaluate_all_models()
        evaluator.generate_confusion_matrices()
        evaluator.save_results()
        
        best_model, best_score = evaluator.get_best_model()
        
        logger.info("Evaluation pipeline completed successfully")
        
        return evaluator
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
