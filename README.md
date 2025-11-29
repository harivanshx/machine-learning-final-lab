# Hotel Reservations ML Pipeline

A modular machine learning pipeline for predicting hotel reservation cancellations.

## Project Structure

```
.
├── data_loader.py           # Data loading and validation
├── preprocess.py            # Data preprocessing and outlier treatment
├── feature_engineering.py   # Feature creation and encoding
├── train.py                 # Model training
├── evaluate.py              # Model evaluation
├── main.py                  # Main pipeline orchestrator
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── models/                 # Saved models (created during execution)
├── results/                # Evaluation results (created during execution)
└── intermediate_data/      # Intermediate data files (optional)
```

## Features

- **Modular Design**: Each component is separated into its own module for easy maintenance
- **Comprehensive Logging**: All operations are logged with timestamps and severity levels
- **Error Handling**: Robust error handling throughout the pipeline
- **Configurable**: Command-line arguments for easy configuration
- **Multiple Models**: Trains and compares Logistic Regression, Random Forest, and XGBoost

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

Run the entire pipeline with default settings:
```bash
python main.py
```

### Command-Line Options

```bash
python main.py --help
```

Available options:
- `--data-path`: Path to the CSV data file (default: 'Hotel Reservations.csv')
- `--outlier-method`: Method for outlier treatment ('cap', 'remove', 'transform')
- `--encoding-method`: Method for categorical encoding ('label', 'onehot')
- `--test-size`: Proportion of test set (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 42)
- `--save-intermediate`: Save intermediate data files
- `--save-models`: Save trained models (default: True)
- `--model-dir`: Directory to save models (default: 'models')
- `--results-dir`: Directory to save results (default: 'results')

### Example Usage

```bash
# Run with custom outlier treatment
python main.py --outlier-method remove

# Run with one-hot encoding and save intermediate files
python main.py --encoding-method onehot --save-intermediate

# Run with custom test size and random state
python main.py --test-size 0.3 --random-state 123
```

### Running Individual Modules

Each module can also be run independently for testing:

```bash
# Test data loader
python data_loader.py

# Test preprocessor
python preprocess.py

# Test feature engineer
python feature_engineering.py

# Test trainer
python train.py

# Test evaluator
python evaluate.py
```

## Pipeline Steps

### 1. Data Loading
- Loads data from CSV file
- Validates data integrity
- Logs dataset information

### 2. Preprocessing
- Identifies numeric columns
- Detects outliers using IQR method
- Treats outliers (capping, removal, or transformation)
- Generates outlier report

### 3. Feature Engineering
- Creates new features:
  - Total nights
  - Total guests
  - Total price
  - Lead time category
  - Booking history score
  - Weekend ratio
  - Price per guest
  - Arrival season
  - And more...
- Encodes categorical variables (label or one-hot encoding)

### 4. Model Training
- Prepares train/test split
- Scales features (for Logistic Regression)
- Trains multiple models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Saves trained models

### 5. Model Evaluation
- Evaluates all models on test set
- Calculates metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- Generates confusion matrices
- Creates classification reports
- Identifies best performing model

## Output

The pipeline generates the following outputs:

### Models Directory
- `logistic_regression.pkl`: Trained Logistic Regression model
- `random_forest.pkl`: Trained Random Forest model
- `xgboost.pkl`: Trained XGBoost model
- `scaler.pkl`: Fitted StandardScaler

### Results Directory
- `model_comparison_results.csv`: Performance metrics for all models
- `confusion_matrices.png`: Confusion matrices visualization
- `*_report.txt`: Classification reports for each model

### Logs
- `pipeline.log`: Complete pipeline execution log

## Logging

The pipeline uses Python's logging module with the following format:
```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

Logs are written to both:
- Console (stdout)
- `pipeline.log` file

## Error Handling

Each module includes comprehensive error handling:
- File not found errors
- Data validation errors
- Model training errors
- Evaluation errors

All errors are logged with detailed messages for debugging.

## Requirements

See `requirements.txt` for a complete list of dependencies.

Main dependencies:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

## CI/CD Integration

This project includes comprehensive CI/CD workflows for automated testing and model retraining.

### GitHub Actions Workflows

**1. ML Pipeline CI/CD** (`.github/workflows/ml-pipeline.yml`)
- Automated testing on push/PR
- Weekly scheduled runs
- Manual trigger with custom parameters
- Performance validation
- Artifact upload (models, results, logs)

**2. Model Retraining** (`.github/workflows/model-retraining.yml`)
- Weekly automated retraining
- Configuration comparison
- Best model selection
- Performance reporting

### Running CI/CD Validation

Test the pipeline before deployment:
```bash
python test_pipeline.py
```

### Documentation

See `CI_CD_GUIDE.md` for detailed CI/CD documentation including:
- Workflow configuration
- Manual triggers
- Performance thresholds
- Troubleshooting
- Integration with other platforms

## Author

Harivansh Bhardwaj


## Github 

https://github.com/harivanshx/machine-learning-final-lab

## License

This project is for educational purposes.
