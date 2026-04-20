"""
Main Pipeline Script for Hospital Readmission Prediction

This script demonstrates the complete end-to-end pipeline:
1. Data preprocessing
2. Model training
3. Model evaluation
4. Prediction on new data
5. Results saving

Usage:
    python main.py --data_path <path> --output_path <path>
"""

import argparse
from pyspark.sql import SparkSession

# Import custom modules
from preprocessing import preprocess_data
from train import prepare_train_test_split, train_all_models, save_model
from evaluation import evaluate_and_compare, save_predictions, save_comparison_results
from predict import predict_pipeline


def main(data_path: str, output_path: str, model_selection_metric: str = 'AUC_ROC'):
    """
    Run the complete pipeline.
    
    Args:
        data_path: Path to raw CSV data
        output_path: Path to save outputs
        model_selection_metric: Metric to use for selecting best model
    """
    print("=" * 70)
    print("HOSPITAL READMISSION PREDICTION PIPELINE")
    print("=" * 70)
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("HospitalReadmissionPrediction") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    print(f"\n✅ Spark session initialized")
    print(f"   Spark version: {spark.version}")
    
    # ========================================
    # STEP 1: DATA PREPROCESSING
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 70)
    
    df, numeric_features, binary_features, categorical_features = preprocess_data(
        spark=spark,
        file_path=data_path
    )
    
    print(f"\n✅ Preprocessing complete")
    print(f"   Total features: {len(numeric_features) + len(binary_features) + len(categorical_features)}")
    
    # ========================================
    # STEP 2: TRAIN/TEST SPLIT
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 2: TRAIN/TEST SPLIT")
    print("=" * 70)
    
    train_df, test_df = prepare_train_test_split(df)
    
    # ========================================
    # STEP 3: MODEL TRAINING
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING")
    print("=" * 70)
    
    results, feature_pipeline = train_all_models(
        train_df=train_df,
        test_df=test_df,
        numeric_features=numeric_features,
        binary_features=binary_features,
        categorical_features=categorical_features
    )
    
    print(f"\n✅ All models trained successfully")
    
    # ========================================
    # STEP 4: MODEL EVALUATION
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 70)
    
    best_model_name, best_model, best_predictions, comparison_df = evaluate_and_compare(
        results=results,
        spark=spark,
        selection_metric=model_selection_metric
    )
    
    print(f"\n✅ Evaluation complete")
    print(f"   Champion model: {best_model_name}")
    
    # ========================================
    # STEP 5: SAVE RESULTS
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 5: SAVE RESULTS")
    print("=" * 70)
    
    # Save best model
    model_path = f"{output_path}/models/readmission_best_model"
    save_model(best_model, model_path)
    
    # Save predictions
    predictions_path = f"{output_path}/predictions/test_predictions"
    save_predictions(best_predictions, predictions_path)
    
    # Save comparison results
    comparison_path = f"{output_path}/results/model_comparison"
    save_comparison_results(comparison_df, comparison_path)
    
    print(f"\n✅ All results saved to: {output_path}")
    
    # ========================================
    # STEP 6: EXAMPLE PREDICTION
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 6: EXAMPLE PREDICTION ON NEW DATA")
    print("=" * 70)
    
    # Use test set as example "new data"
    new_predictions = predict_pipeline(
        spark=spark,
        model_path=model_path,
        data=test_df,
        output_path=f"{output_path}/predictions/new_predictions",
        high_risk_threshold=0.5
    )
    
    # ========================================
    # PIPELINE COMPLETE
    # ========================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Champion Model: {best_model_name}")
    print(f"   Model saved: {model_path}")
    print(f"   Predictions saved: {predictions_path}")
    print(f"   Comparison saved: {comparison_path}")
    print("\n✅ All tasks completed successfully!")
    
    # Stop Spark
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hospital Readmission Prediction Pipeline"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to raw CSV data file"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save outputs (models, predictions, results)"
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        default="AUC_ROC",
        choices=["AUC_ROC", "AUC_PR", "F1_Score", "Accuracy"],
        help="Metric to use for selecting best model (default: AUC_ROC)"
    )
    
    args = parser.parse_args()
    
    main(
        data_path=args.data_path,
        output_path=args.output_path,
        model_selection_metric=args.metric
    )
