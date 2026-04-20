"""
Prediction Module for Hospital Readmission Prediction

This module handles:
- Loading trained models
- Making predictions on new data
- Extracting prediction probabilities
- Saving prediction results
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vector, Vectors
from typing import Optional


def load_model(spark: SparkSession, model_path: str) -> PipelineModel:
    """
    Load a trained model from disk.
    
    Args:
        spark: SparkSession instance
        model_path: Path to saved model
        
    Returns:
        Loaded PipelineModel
    """
    print(f"Loading model from: {model_path}")
    model = PipelineModel.load(model_path)
    print("✅ Model loaded successfully")
    return model


def extract_probability_udf():
    """
    Create UDF to extract probability for positive class.
    
    Returns:
        UDF function
    """
    def extract_prob(probability_vector):
        """Extract probability of positive class (index 1)."""
        if probability_vector is not None:
            return float(probability_vector[1])
        return None
    
    return udf(extract_prob, DoubleType())


def make_predictions(model: PipelineModel, 
                    data: DataFrame,
                    include_probability: bool = True) -> DataFrame:
    """
    Make predictions on new data.
    
    Args:
        model: Trained PipelineModel
        data: DataFrame with features
        include_probability: Whether to extract probability scores
        
    Returns:
        DataFrame with predictions
    """
    print("Making predictions...")
    predictions = model.transform(data)
    
    if include_probability:
        # Extract probability for positive class
        prob_udf = extract_probability_udf()
        predictions = predictions.withColumn(
            'readmission_probability',
            prob_udf(col('probability'))
        )
    
    print(f"✅ Predictions generated for {predictions.count():,} records")
    return predictions


def get_prediction_summary(predictions: DataFrame, 
                          target_col: Optional[str] = None) -> DataFrame:
    """
    Get summary statistics of predictions.
    
    Args:
        predictions: DataFrame with predictions
        target_col: Optional actual label column for comparison
        
    Returns:
        Summary DataFrame
    """
    from pyspark.sql.functions import count, sum as spark_sum, avg, round as spark_round
    
    summary = predictions.groupBy('prediction').agg(
        count('*').alias('count'),
        round(avg('readmission_probability') * 100, 2).alias('avg_probability_pct')
    )
    
    if target_col and target_col in predictions.columns:
        # Add accuracy by prediction group
        summary = predictions.groupBy('prediction').agg(
            count('*').alias('count'),
            round(avg('readmission_probability') * 100, 2).alias('avg_probability_pct'),
            round(avg(col(target_col)) * 100, 2).alias('actual_readmit_rate_pct')
        )
    
    print("\nPrediction Summary:")
    summary.show()
    
    return summary


def filter_high_risk_patients(predictions: DataFrame, 
                              threshold: float = 0.5) -> DataFrame:
    """
    Filter patients with high readmission risk.
    
    Args:
        predictions: DataFrame with predictions
        threshold: Probability threshold for high risk
        
    Returns:
        DataFrame with high-risk patients
    """
    high_risk = predictions.filter(col('readmission_probability') >= threshold)
    
    count = high_risk.count()
    total = predictions.count()
    pct = (count / total) * 100 if total > 0 else 0
    
    print(f"\nHigh-risk patients (probability >= {threshold}):")
    print(f"   Count: {count:,} ({pct:.2f}% of total)")
    
    return high_risk


def save_predictions(predictions: DataFrame, 
                    output_path: str,
                    columns: Optional[list] = None):
    """
    Save predictions to parquet format.
    
    Args:
        predictions: DataFrame with predictions
        output_path: Path to save predictions
        columns: Optional list of columns to save (default: key columns only)
    """
    if columns is None:
        # Default columns to save
        columns = [
            'encounter_id', 'patient_nbr', 'prediction', 
            'readmission_probability', 'probability'
        ]
        # Add target column if it exists
        if 'readmit_30' in predictions.columns:
            columns.insert(2, 'readmit_30')
    
    # Select only existing columns
    available_cols = [c for c in columns if c in predictions.columns]
    
    predictions.select(available_cols).write.mode("overwrite").parquet(output_path)
    print(f"✅ Predictions saved to: {output_path}")


def predict_pipeline(spark: SparkSession,
                    model_path: str,
                    data: DataFrame,
                    output_path: Optional[str] = None,
                    high_risk_threshold: float = 0.5) -> DataFrame:
    """
    Complete prediction pipeline.
    
    Args:
        spark: SparkSession instance
        model_path: Path to trained model
        data: DataFrame with features
        output_path: Optional path to save predictions
        high_risk_threshold: Threshold for high-risk classification
        
    Returns:
        DataFrame with predictions
    """
    print("=" * 60)
    print("PREDICTION PIPELINE")
    print("=" * 60)
    
    # Load model
    model = load_model(spark, model_path)
    
    # Make predictions
    predictions = make_predictions(model, data, include_probability=True)
    
    # Get summary
    get_prediction_summary(predictions, target_col='readmit_30')
    
    # Identify high-risk patients
    high_risk = filter_high_risk_patients(predictions, high_risk_threshold)
    
    # Save if output path provided
    if output_path:
        save_predictions(predictions, output_path)
    
    print("=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)
    
    return predictions
