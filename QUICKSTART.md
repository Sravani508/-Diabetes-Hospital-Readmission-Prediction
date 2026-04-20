# Quick Start Guide

Get up and running with the Hospital Readmission Prediction project in minutes!

## 🚀 5-Minute Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/yourusername/diabetes-readmission-prediction.git
cd diabetes-readmission-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get the Data

Download the dataset from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008) and place `diabetic_data.csv` in your data directory.

### 3. Run the Pipeline

```python
from pyspark.sql import SparkSession
from src.main import main

# Run complete pipeline
main(
    data_path="path/to/diabetic_data.csv",
    output_path="output/"
)
```

That's it! Your models will be trained and saved to the output directory.

## 📚 Common Use Cases

### Use Case 1: Train Models

```python
from pyspark.sql import SparkSession
from src.preprocessing import preprocess_data
from src.train import prepare_train_test_split, train_all_models

# Initialize Spark
spark = SparkSession.builder.appName("Readmission").getOrCreate()

# Preprocess data
df, num_feat, bin_feat, cat_feat = preprocess_data(
    spark, "data/diabetic_data.csv"
)

# Split and train
train_df, test_df = prepare_train_test_split(df)
results, pipeline = train_all_models(
    train_df, test_df, num_feat, bin_feat, cat_feat
)
```

### Use Case 2: Evaluate Models

```python
from src.evaluation import evaluate_and_compare

# Compare all models
best_name, best_model, predictions, comparison = evaluate_and_compare(
    results, spark, selection_metric='AUC_ROC'
)

# View comparison
comparison.show()
```

### Use Case 3: Make Predictions

```python
from src.predict import predict_pipeline

# Load model and predict
predictions = predict_pipeline(
    spark=spark,
    model_path="output/models/readmission_best_model",
    data=new_patient_data,
    high_risk_threshold=0.5
)

# View high-risk patients
high_risk = predictions.filter("readmission_probability >= 0.5")
high_risk.select("patient_nbr", "readmission_probability").show()
```

### Use Case 4: Custom Feature Engineering

```python
from src.preprocessing import (
    load_data, 
    create_target_variable,
    create_admission_features,
    create_utilization_features
)

# Load and process step by step
df = load_data(spark, "data/diabetic_data.csv")
df = create_target_variable(df)
df = create_admission_features(df)
df = create_utilization_features(df)

# Add your custom features here
df = df.withColumn("my_custom_feature", ...)
```

## 🎯 Key Functions Reference

### Preprocessing (`src/preprocessing.py`)

```python
# Complete preprocessing pipeline
df, num_feat, bin_feat, cat_feat = preprocess_data(spark, file_path)

# Individual functions
df = replace_missing_indicators(df)
df = remove_invalid_records(df)
df = create_admission_features(df)
df = create_utilization_features(df)
df = create_age_features(df)
```

### Training (`src/train.py`)

```python
# Split data
train_df, test_df = prepare_train_test_split(df)

# Calculate class weights
weight_neg, weight_pos = calculate_class_weights(train_df)

# Train all models
results, pipeline = train_all_models(
    train_df, test_df, num_feat, bin_feat, cat_feat
)

# Save model
save_model(best_model, "path/to/save")
```

### Evaluation (`src/evaluation.py`)

```python
# Evaluate single model
metrics = evaluate_model(predictions, "Model Name")

# Compare all models
comparison_df = compare_models(results, spark)

# Select best model
best_name, best_model, predictions = select_best_model(
    comparison_df, results, metric='AUC_ROC'
)

# Save results
save_predictions(predictions, "path/to/predictions")
save_comparison_results(comparison_df, "path/to/comparison")
```

### Prediction (`src/predict.py`)

```python
# Load model
model = load_model(spark, "path/to/model")

# Make predictions
predictions = make_predictions(model, data)

# Get summary
summary = get_prediction_summary(predictions)

# Filter high-risk
high_risk = filter_high_risk_patients(predictions, threshold=0.5)

# Complete pipeline
predictions = predict_pipeline(
    spark, model_path, data, output_path, threshold
)
```

## 🔧 Configuration

### Adjust Model Parameters

Edit parameters in `src/train.py`:

```python
# Random Forest
rf = RandomForestClassifier(
    numTrees=100,      # Increase for better performance
    maxDepth=10,       # Increase for more complex patterns
    minInstancesPerNode=5
)

# Logistic Regression
lr = LogisticRegression(
    maxIter=100,
    regParam=0.01,     # Regularization strength
    elasticNetParam=0.5  # L1/L2 mix
)
```

### Change Feature Selection

Modify feature lists in `src/preprocessing.py`:

```python
numeric_features = [
    'time_in_hospital',
    'num_medications',
    # Add your features here
]
```

### Adjust Thresholds

```python
# High-risk threshold
high_risk = predictions.filter("readmission_probability >= 0.3")

# Missing value threshold
MISSING_THRESHOLD = 40.0  # Drop columns with >40% missing
```

## 📊 Output Files

After running the pipeline, you'll find:

```
output/
├── models/
│   └── readmission_best_model/    # Trained model
├── predictions/
│   ├── test_predictions/          # Test set predictions
│   └── new_predictions/           # New data predictions
└── results/
    └── model_comparison/          # Performance comparison
```

## 🐛 Troubleshooting

### Issue: Spark not found

```bash
pip install pyspark
```

### Issue: Out of memory

```python
# Reduce data size or increase Spark memory
spark = SparkSession.builder \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
```

### Issue: Missing columns

Ensure your data has all required columns. Check `data/README.md` for schema.

### Issue: Model loading fails

Verify the model path exists and contains the complete model directory.

## 💡 Tips

* **Start small**: Test with a sample of data first
* **Monitor memory**: Large datasets may require cluster computing
* **Tune hyperparameters**: Default values are a starting point
* **Validate features**: Check feature distributions before training
* **Save checkpoints**: Save intermediate results during long runs

## 📖 Next Steps

* Read the full [README.md](README.md) for detailed documentation
* Check [data/README.md](data/README.md) for dataset details
* Review [models/README.md](models/README.md) for deployment guide

## 🆘 Getting Help

* Check existing [GitHub Issues](https://github.com/yourusername/repo/issues)
* Review documentation in README files
* Open a new issue with details about your problem

---

Happy predicting! 🎉
