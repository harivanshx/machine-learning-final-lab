"""
Test Script for CI/CD Pipeline
Validates that all components work correctly before deployment
"""

import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineValidator:
    """Validates the ML pipeline for CI/CD"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def validate_file_structure(self) -> bool:
        """Validate required files exist"""
        logger.info("Validating file structure...")
        
        required_files = [
            'data_loader.py',
            'preprocess.py',
            'feature_engineering.py',
            'train.py',
            'evaluate.py',
            'main.py',
            'requirements.txt',
            'README.md'
        ]
        
        for file in required_files:
            if not Path(file).exists():
                self.errors.append(f"Missing required file: {file}")
            else:
                logger.info(f"✓ Found: {file}")
        
        return len(self.errors) == 0
    
    def validate_imports(self) -> bool:
        """Validate all required packages can be imported"""
        logger.info("Validating imports...")
        
        required_packages = [
            ('pandas', 'pd'),
            ('numpy', 'np'),
            ('sklearn', None),
            ('xgboost', None),
            ('matplotlib', None),
            ('seaborn', None)
        ]
        
        for package, alias in required_packages:
            try:
                if alias:
                    exec(f"import {package} as {alias}")
                else:
                    exec(f"import {package}")
                logger.info(f"✓ Imported: {package}")
            except ImportError as e:
                self.errors.append(f"Failed to import {package}: {e}")
        
        return len(self.errors) == 0
    
    def validate_data_file(self) -> bool:
        """Validate data file exists and is readable"""
        logger.info("Validating data file...")
        
        data_file = 'Hotel Reservations.csv'
        
        if not Path(data_file).exists():
            self.errors.append(f"Data file not found: {data_file}")
            return False
        
        try:
            import pandas as pd
            df = pd.read_csv(data_file, nrows=5)
            logger.info(f"✓ Data file readable: {data_file}")
            logger.info(f"  Columns: {len(df.columns)}")
            return True
        except Exception as e:
            self.errors.append(f"Failed to read data file: {e}")
            return False
    
    def validate_modules(self) -> bool:
        """Validate all modules can be imported"""
        logger.info("Validating modules...")
        
        modules = [
            'data_loader',
            'preprocess',
            'feature_engineering',
            'train',
            'evaluate'
        ]
        
        for module in modules:
            try:
                exec(f"import {module}")
                logger.info(f"✓ Module: {module}")
            except Exception as e:
                self.errors.append(f"Failed to import {module}: {e}")
        
        return len(self.errors) == 0
    
    def validate_pipeline_execution(self) -> bool:
        """Validate pipeline can execute"""
        logger.info("Validating pipeline execution...")
        
        try:
            # Import main module
            import main
            
            # Check if MLPipeline class exists
            if not hasattr(main, 'MLPipeline'):
                self.errors.append("MLPipeline class not found in main.py")
                return False
            
            logger.info("✓ Pipeline structure valid")
            return True
            
        except Exception as e:
            self.errors.append(f"Pipeline validation failed: {e}")
            return False
    
    def validate_model_saving(self) -> bool:
        """Validate models directory structure"""
        logger.info("Validating model saving capability...")
        
        models_dir = Path('models')
        
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pkl'))
            if model_files:
                logger.info(f"✓ Found {len(model_files)} model files")
                for model_file in model_files:
                    logger.info(f"  - {model_file.name}")
                return True
            else:
                self.warnings.append("No model files found (run pipeline first)")
        else:
            self.warnings.append("Models directory not found (run pipeline first)")
        
        return True
    
    def validate_results_generation(self) -> bool:
        """Validate results directory structure"""
        logger.info("Validating results generation...")
        
        results_dir = Path('results')
        
        if results_dir.exists():
            result_files = list(results_dir.glob('*'))
            if result_files:
                logger.info(f"✓ Found {len(result_files)} result files")
                for result_file in result_files:
                    logger.info(f"  - {result_file.name}")
                return True
            else:
                self.warnings.append("No result files found (run pipeline first)")
        else:
            self.warnings.append("Results directory not found (run pipeline first)")
        
        return True
    
    def run_all_validations(self) -> bool:
        """Run all validations"""
        logger.info("="*80)
        logger.info("Starting CI/CD Pipeline Validation")
        logger.info("="*80)
        
        validations = [
            self.validate_file_structure,
            self.validate_imports,
            self.validate_data_file,
            self.validate_modules,
            self.validate_pipeline_execution,
            self.validate_model_saving,
            self.validate_results_generation
        ]
        
        all_passed = True
        
        for validation in validations:
            try:
                if not validation():
                    all_passed = False
            except Exception as e:
                self.errors.append(f"Validation error: {e}")
                all_passed = False
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("Validation Summary")
        logger.info("="*80)
        
        if self.errors:
            logger.error(f"\n❌ {len(self.errors)} Error(s):")
            for error in self.errors:
                logger.error(f"  - {error}")
        
        if self.warnings:
            logger.warning(f"\n⚠️  {len(self.warnings)} Warning(s):")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        if all_passed and not self.errors:
            logger.info("\n✅ All validations passed!")
            logger.info("Pipeline is ready for CI/CD deployment")
            return True
        else:
            logger.error("\n❌ Validation failed!")
            logger.error("Please fix the errors before deploying")
            return False


def main():
    """Main entry point"""
    validator = PipelineValidator()
    
    success = validator.run_all_validations()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
