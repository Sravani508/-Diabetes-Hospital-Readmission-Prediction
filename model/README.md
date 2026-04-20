# Models Directory

## Overview

This directory contains trained machine learning models for hospital readmission prediction. Models are saved in PySpark's native format for easy loading and deployment.

## Model Storage

### Directory Structure

```
models/
├── readmission_best_model/        # Best performing model (Random Forest)
│   ├── metadata/
│   └── stages/
├── test_predictions/              # Predictions on test set
└── model_comparison/              # Performance comparison results
```

### Model Format

Models are saved using PySpark's `PipelineModel.write().save()` method, which creates a directory containing:
* **metadata/**: Model configuration and parameters
* **stages/**: Individual pipeline stages (encoders, scalers, classifier)

**Note**: Models are saved as directories, not single files. The entire directory must be preserved for loading.

## Trained Models

### 1. Random Forest (Champion Model)

**Performance:**
* AUC-ROC: 0.6494
* AUC-PR: 0.2156
* F1 Score: 0.5891
* Accuracy: 0.6145

**Hyperparameters:**
* Number of trees: 100
* Max depth: 10
* Min instances per node: 5
* Feature subset strategy: auto

**Use Case:** Best overall performance, recommended for production deployment.

**Performance Visualizations:**

See the [main README](../README.md#-results) for detailed performance visualizations including:
* ROC Curve showing AUC of 0.6494
* Precision-Recall Curve with AUC-PR of 0.2156
* Confusion Matrix breakdown
* Feature importance rankings

All visualization images are available in the `images/` directory.

### 2. Gradient Boosted Trees

**Performance:**
* AUC-ROC: 0.6441
* AUC-PR: 0.2089
* F1 Score: 0.5856
* Accuracy: 0.6104

**Hyperparameters:**
* Max iterations: 100
* Max depth: 5
* Step size: 0.1

**Use Case:** Good alternative with slightly faster inference time.

### 3. Logistic Regression

**Performance:**
* AUC-ROC: 0.6389
* AUC-PR: 0.2034
* F1 Score: 0.5823
* Accuracy: 0.6071

**Hyperparameters:**
* Max iterations: 100
* Regularization: 0.01
* ElasticNet mixing: 0.5

**Use Case:** Fastest inference, interpretable coefficients, baseline model.

## Loading Models

### Using Python

```python
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder.appName("Readmission").getOrCreate()

# Load model
model_path = "models/readmission_best_model"
model = PipelineModel.load(model_path)

# Make predictions
predictions = model.transform(new_data)
```

### Using the predict.py Module

```python
from src.predict import predict_pipeline

predictions = predict_pipeline(
    spark=spark,
    model_path="models/readmission_best_model",
    data=new_data_df,
    output_path="output/predictions",
    high_risk_threshold=0.5
)
```

## Model Inputs

### Required Features

The model expects a DataFrame with the following features:

**Numeric Features (11):**
* `time_in_hospital`
* `num_lab_procedures`
* `num_procedures`
* `num_medications`
* `number_outpatient`
* `number_emergency`
* `number_inpatient`
* `number_diagnoses`
* `total_prior_visits`
* `complexity_score`
* `age_numeric`

**Binary Features (12):**
* `emergency_admission`
* `admitted_from_er`
* `discharged_not_home`
* `left_ama`
* `high_utilizer`
* `frequent_er_visitor`
* `is_elderly`
* `a1c_tested`
* `a1c_abnormal`
* `glucose_tested`
* `on_insulin`
* `med_changed`

**Categorical Features (5):**
* `race`
* `gender`
* `admission_type_group`
* `admission_source_group`
* `discharge_group`

### Feature Preprocessing

The model pipeline includes:
1. **String Indexing**: Converts categorical strings to numeric indices
2. **One-Hot Encoding**: Creates binary vectors for categorical features
3. **Vector Assembly**: Combines all features into a single vector
4. **Standard Scaling**: Normalizes numeric features (mean=0, std=1)

**Note**: All preprocessing is included in the saved model pipeline. You only need to provide raw features.

## Model Outputs

### Prediction Columns

* `prediction`: Binary prediction (0 = not readmitted, 1 = readmitted within 30 days)
* `probability`: Vector of probabilities [P(class 0), P(class 1)]
* `readmission_probability`: Extracted probability of readmission (class 1)
* `rawPrediction`: Raw prediction scores before probability calibration

### Interpreting Predictions

* **Probability < 0.3**: Low risk of readmission
* **Probability 0.3-0.5**: Moderate risk
* **Probability > 0.5**: High risk (default threshold)
* **Probability > 0.7**: Very high risk

**Recommendation**: Adjust threshold based on your use case:
* **High recall** (catch more readmissions): Lower threshold (e.g., 0.3)
* **High precision** (fewer false alarms): Higher threshold (e.g., 0.6)

## Model Versioning

### Version Naming Convention

```
readmission_model_v{version}_{date}_{metric}
```

Example: `readmission_model_v1_20260205_auc0.6494`

### Tracking Model Versions

Consider using MLflow or similar tools for production deployments:

```python
import mlflow
import mlflow.spark

# Log model
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("num_trees", 100)
    mlflow.log_metric("auc_roc", 0.6494)
    mlflow.spark.log_model(model, "model")
```

## Model Retraining

### When to Retrain

* **Scheduled**: Every 6-12 months with new data
* **Performance Degradation**: If AUC-ROC drops below 0.60
* **Data Drift**: Significant changes in patient demographics or clinical practices
* **New Features**: When additional data sources become available

### Retraining Process

1. Collect new data (maintain same schema)
2. Run preprocessing pipeline: `src/preprocessing.py`
3. Train models: `src/train.py`
4. Evaluate performance: `src/evaluation.py`
5. Compare with current champion model
6. Deploy if performance improves

## Model Deployment

### Batch Predictions

```python
# Load model once
model = PipelineModel.load("models/readmission_best_model")

# Process data in batches
for batch in data_batches:
    predictions = model.transform(batch)
    predictions.write.mode("append").parquet("output/predictions")
```

### Real-Time Predictions

For real-time scoring, consider:
* **Databricks Model Serving**: Deploy model as REST API
* **MLflow Model Registry**: Version control and staging
* **Spark Structured Streaming**: Process streaming data

### Production Checklist

- [ ] Model validated on holdout test set
- [ ] Performance metrics documented
- [ ] Feature engineering pipeline tested
- [ ] Model versioning implemented
- [ ] Monitoring and alerting configured
- [ ] Rollback plan prepared
- [ ] Documentation updated

## Model Limitations

1. **Temporal Scope**: Trained on 1999-2008 data; may not reflect current clinical practices
2. **Geographic Scope**: US hospitals only; may not generalize internationally
3. **Class Imbalance**: Optimized for imbalanced data (11% positive class)
4. **Feature Availability**: Requires complete feature set for accurate predictions
5. **Interpretability**: Tree-based models are less interpretable than logistic regression

## Support

For questions about model usage or deployment:
1. Check the main README.md
2. Review the source code in `src/`
3. Open an issue on GitHub

---

**Last Updated**: February 2026
