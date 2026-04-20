"""
Model Evaluation Module for Hospital Readmission Prediction

This comprehensive module handles:
- Model performance evaluation (AUC-ROC, AUC-PR, F1, Accuracy, Precision, Recall)
- Model comparison and selection
- Confusion matrix analysis with sensitivity/specificity
- Prediction distribution analysis
- Threshold testing
- Data leakage validation
- Visualization generation
- Results saving (parquet files)
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when, avg, count, sum as spark_sum, lit, expr, udf, desc
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np


TARGET_COLUMN = "readmit_30"


# ============================================================
# SECTION 1: BASIC EVALUATION FUNCTIONS
# ============================================================

def create_evaluators() -> Dict[str, object]:
    """
    Create all evaluation metrics.
    
    Returns:
        Dictionary of evaluator objects
    """
    evaluators = {
        'auc_roc': BinaryClassificationEvaluator(
            labelCol=TARGET_COLUMN,
            rawPredictionCol='rawPrediction',
            metricName='areaUnderROC'
        ),
        'auc_pr': BinaryClassificationEvaluator(
            labelCol=TARGET_COLUMN,
            rawPredictionCol='rawPrediction',
            metricName='areaUnderPR'
        ),
        'accuracy': MulticlassClassificationEvaluator(
            labelCol=TARGET_COLUMN,
            predictionCol='prediction',
            metricName='accuracy'
        ),
        'f1': MulticlassClassificationEvaluator(
            labelCol=TARGET_COLUMN,
            predictionCol='prediction',
            metricName='f1'
        ),
        'precision': MulticlassClassificationEvaluator(
            labelCol=TARGET_COLUMN,
            predictionCol='prediction',
            metricName='weightedPrecision'
        ),
        'recall': MulticlassClassificationEvaluator(
            labelCol=TARGET_COLUMN,
            predictionCol='prediction',
            metricName='weightedRecall'
        )
    }
    return evaluators


def evaluate_model(predictions: DataFrame, model_name: str, 
                  evaluators: Dict[str, object] = None) -> Dict[str, float]:
    """
    Evaluate a single model with all metrics.
    
    Args:
        predictions: DataFrame with predictions
        model_name: Name of the model
        evaluators: Optional pre-created evaluators (for efficiency)
    
    Returns:
        Dictionary of metric scores
    """
    if evaluators is None:
        evaluators = create_evaluators()
    
    print(f"\n--- Evaluating {model_name} ---")
    
    metrics = {'model': model_name}
    
    for metric_name, evaluator in evaluators.items():
        try:
            score = evaluator.evaluate(predictions)
            metrics[metric_name] = round(score, 4)
            print(f"{metric_name.upper().replace('_', '-')}: {score:.4f}")
        except Exception as e:
            print(f"Warning: Could not calculate {metric_name} for {model_name}: {e}")
            metrics[metric_name] = None
    
    return metrics


def calculate_confusion_matrix(predictions: DataFrame) -> Dict[str, int]:
    """
    Calculate confusion matrix components.
    
    Args:
        predictions: DataFrame with predictions and actual labels
    
    Returns:
        Dictionary with tp, tn, fp, fn counts (lowercase keys)
    """
    tp = predictions.filter((col(TARGET_COLUMN) == 1) & (col('prediction') == 1)).count()
    tn = predictions.filter((col(TARGET_COLUMN) == 0) & (col('prediction') == 0)).count()
    fp = predictions.filter((col(TARGET_COLUMN) == 0) & (col('prediction') == 1)).count()
    fn = predictions.filter((col(TARGET_COLUMN) == 1) & (col('prediction') == 0)).count()
    
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def print_confusion_matrix(cm: Dict[str, int], model_name: str) -> None:
    """
    Print formatted confusion matrix.
    
    Args:
        cm: Confusion matrix dictionary (supports both 'tp'/'TP' keys)
        model_name: Name of the model
    """
    # Handle both lowercase and uppercase keys
    tp = cm.get('tp', cm.get('TP', 0))
    tn = cm.get('tn', cm.get('TN', 0))
    fp = cm.get('fp', cm.get('FP', 0))
    fn = cm.get('fn', cm.get('FN', 0))
    
    print(f"\n{model_name} - Confusion Matrix:")
    print("                 Predicted")
    print("                 0      1")
    print(f"Actual 0    {tn:6d} {fp:6d}")
    print(f"       1    {fn:6d} {tp:6d}")
    
    # Calculate additional metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


# ============================================================
# SECTION 2: ADVANCED ANALYSIS FUNCTIONS
# ============================================================

def analyze_confusion_matrix(predictions: DataFrame, model_name: str) -> Tuple[Dict, float, float]:
    """
    Analyze and display confusion matrix with sensitivity/specificity.
    
    Args:
        predictions: DataFrame with predictions
        model_name: Name of the model
    
    Returns:
        Tuple of (confusion_matrix_dict, sensitivity, specificity)
    """
    print("=" * 60)
    print(f"CONFUSION MATRIX: {model_name}")
    print("=" * 60)
    
    # Display prediction vs actual
    print("\nPrediction vs Actual:")
    predictions.groupBy(TARGET_COLUMN, 'prediction').count().orderBy(TARGET_COLUMN, 'prediction').show()
    
    # Calculate confusion matrix components
    cm = calculate_confusion_matrix(predictions)
    
    print(f"\nConfusion Matrix Breakdown:")
    print(f"   True Positives (correctly predicted readmission): {cm['tp']:,}")
    print(f"   True Negatives (correctly predicted no readmission): {cm['tn']:,}")
    print(f"   False Positives (incorrectly predicted readmission): {cm['fp']:,}")
    print(f"   False Negatives (missed readmissions): {cm['fn']:,}")
    
    # Calculate sensitivity and specificity
    sensitivity = cm['tp'] / (cm['tp'] + cm['fn']) if (cm['tp'] + cm['fn']) > 0 else 0
    specificity = cm['tn'] / (cm['tn'] + cm['fp']) if (cm['tn'] + cm['fp']) > 0 else 0
    
    print(f"\nAdditional Metrics:")
    print(f"   Sensitivity (Recall for positive class): {sensitivity:.4f}")
    print(f"   Specificity (Recall for negative class): {specificity:.4f}")
    
    return cm, sensitivity, specificity


def analyze_prediction_distribution(predictions: DataFrame) -> DataFrame:
    """
    Analyze prediction probability distribution.
    
    Args:
        predictions: DataFrame with predictions and probabilities
    
    Returns:
        DataFrame with probability column added
    """
    print("\n" + "=" * 60)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Extract probability of positive class
    def extract_prob(v, index):
        try:
            return float(v[index])
        except:
            return None
    
    extract_prob_udf = udf(lambda v: extract_prob(v, 1), DoubleType())
    
    predictions_detailed = predictions.withColumn(
        'probability_readmit',
        extract_prob_udf(col('probability'))
    )
    
    # Show prediction distribution
    print("\n--- Prediction Score Distribution ---")
    predictions_detailed.select('probability_readmit').describe().show()
    
    # Distribution by actual class
    print("\n--- Probability Distribution by Actual Class ---")
    predictions_detailed.groupBy(TARGET_COLUMN).agg(
        avg('probability_readmit').alias('mean_prob'),
        expr('percentile_approx(probability_readmit, 0.25)').alias('p25'),
        expr('percentile_approx(probability_readmit, 0.50)').alias('median'),
        expr('percentile_approx(probability_readmit, 0.75)').alias('p75'),
        count('*').alias('count')
    ).show()
    
    return predictions_detailed


def get_roc_pr_data(predictions: DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract data for ROC and Precision-Recall curves.
    
    Args:
        predictions: DataFrame with predictions and probabilities
    
    Returns:
        Tuple of (y_true, y_prob, y_pred) as numpy arrays
    """
    # Extract probability of positive class
    def extract_prob(v, index):
        try:
            return float(v[index])
        except:
            return None
    
    extract_prob_udf = udf(lambda v: extract_prob(v, 1), DoubleType())
    
    predictions_with_prob = predictions.withColumn(
        'probability_readmit',
        extract_prob_udf(col('probability'))
    )
    
    # Convert to pandas for sklearn metrics
    predictions_pd = predictions_with_prob.select(
        TARGET_COLUMN,
        'prediction',
        'probability_readmit'
    ).toPandas()
    
    y_true = predictions_pd[TARGET_COLUMN].values
    y_prob = predictions_pd['probability_readmit'].values
    y_pred = predictions_pd['prediction'].values
    
    return y_true, y_prob, y_pred


def test_model_at_thresholds(predictions: DataFrame, 
                            thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]) -> None:
    """
    Test model performance at different probability thresholds.
    
    Args:
        predictions: DataFrame with predictions and probabilities
        thresholds: List of threshold values to test
    """
    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)
    
    # Extract probability
    def extract_prob(v, index):
        try:
            return float(v[index])
        except:
            return None
    
    extract_prob_udf = udf(lambda v: extract_prob(v, 1), DoubleType())
    
    predictions_with_prob = predictions.withColumn(
        'probability_readmit',
        extract_prob_udf(col('probability'))
    )
    
    print("\nPerformance at Different Thresholds:")
    print(f"{'Threshold':<12} {'TP':<8} {'TN':<8} {'FP':<8} {'FN':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 100)
    
    for threshold in thresholds:
        # Apply threshold
        preds_at_threshold = predictions_with_prob.withColumn(
            'pred_at_threshold',
            when(col('probability_readmit') >= threshold, 1).otherwise(0)
        )
        
        # Calculate metrics
        tp = preds_at_threshold.filter((col(TARGET_COLUMN) == 1) & (col('pred_at_threshold') == 1)).count()
        tn = preds_at_threshold.filter((col(TARGET_COLUMN) == 0) & (col('pred_at_threshold') == 0)).count()
        fp = preds_at_threshold.filter((col(TARGET_COLUMN) == 0) & (col('pred_at_threshold') == 1)).count()
        fn = preds_at_threshold.filter((col(TARGET_COLUMN) == 1) & (col('pred_at_threshold') == 0)).count()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{threshold:<12.2f} {tp:<8} {tn:<8} {fp:<8} {fn:<8} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")


def validate_data_leakage_prevention(train_df: DataFrame, test_df: DataFrame, 
                                    patient_id_col: str = 'patient_nbr') -> bool:
    """
    Validate that there is no patient overlap between train and test sets.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        patient_id_col: Name of patient ID column
    
    Returns:
        True if no overlap, False otherwise
    """
    print("\n" + "=" * 60)
    print("DATA LEAKAGE VALIDATION")
    print("=" * 60)
    
    train_patients = set(train_df.select(patient_id_col).distinct().toPandas()[patient_id_col].tolist())
    test_patients = set(test_df.select(patient_id_col).distinct().toPandas()[patient_id_col].tolist())
    
    overlap = train_patients.intersection(test_patients)
    
    print(f"\nTrain patients: {len(train_patients):,}")
    print(f"Test patients: {len(test_patients):,}")
    print(f"Overlapping patients: {len(overlap):,}")
    
    if len(overlap) == 0:
        print("\n✅ PASS: No patient overlap detected - data leakage prevented!")
        return True
    else:
        print(f"\n❌ FAIL: {len(overlap)} patients appear in both train and test sets!")
        print("   This indicates data leakage - fix the train/test split logic.")
        return False


# ============================================================
# SECTION 3: MODEL COMPARISON FUNCTIONS
# ============================================================

def compare_models(results: Dict[str, Tuple], spark: SparkSession, 
                  evaluators: Dict[str, object] = None) -> DataFrame:
    """
    Compare all models and create comparison DataFrame.
    
    Args:
        results: Dictionary mapping model names to (model, predictions) tuples
        spark: SparkSession instance
        evaluators: Optional pre-created evaluators
    
    Returns:
        DataFrame with model comparison metrics
    """
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    if evaluators is None:
        evaluators = create_evaluators()
    
    comparison_data = []
    
    for model_name, (model, predictions) in results.items():
        print(f"\nEvaluating {model_name}...")
        
        # Get metrics
        metrics = evaluate_model(predictions, model_name, evaluators)
        
        # Get confusion matrix
        cm = calculate_confusion_matrix(predictions)
        print_confusion_matrix(cm, model_name)
        
        # Add to comparison
        comparison_data.append({
            'Model': model_name,
            'AUC_ROC': metrics['auc_roc'],
            'AUC_PR': metrics['auc_pr'],
            'Accuracy': metrics['accuracy'],
            'F1_Score': metrics['f1'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'True_Positives': cm['tp'],
            'True_Negatives': cm['tn'],
            'False_Positives': cm['fp'],
            'False_Negatives': cm['fn']
        })
    
    # Create comparison DataFrame
    comparison_df = spark.createDataFrame(comparison_data)
    
    print("\n" + "=" * 60)
    print("SUMMARY - All Models")
    print("=" * 60)
    comparison_df.select('Model', 'AUC_ROC', 'AUC_PR', 'F1_Score', 'Accuracy').show(truncate=False)
    
    return comparison_df


def select_best_model(comparison_df: DataFrame, 
                     results: Dict[str, Tuple],
                     metric: str = 'AUC_ROC') -> Tuple[str, PipelineModel, DataFrame]:
    """
    Select the best model based on specified metric.
    
    Args:
        comparison_df: DataFrame with model comparison
        results: Dictionary of model results
        metric: Metric to use for selection (default: AUC_ROC)
    
    Returns:
        Tuple of (best_model_name, best_model, best_predictions)
    """
    # Find best model
    best_row = comparison_df.orderBy(col(metric).desc()).first()
    best_model_name = best_row['Model']
    best_score = best_row[metric]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   {metric}: {best_score:.4f}")
    
    best_model, best_predictions = results[best_model_name]
    
    return best_model_name, best_model, best_predictions


# ============================================================
# SECTION 4: VISUALIZATION GENERATION
# ============================================================

def generate_test_visualizations(test_results: Dict, output_dir: str = "images",
                                feature_names: List[str] = None, 
                                feature_importance: np.ndarray = None) -> None:
    """
    Generate all test visualizations and save to images folder.
    
    Args:
        test_results: Dictionary from run_comprehensive_tests()
        output_dir: Directory to save images (default: "images")
        feature_names: Optional list of feature names for importance plot
        feature_importance: Optional array of feature importance values
    """
    print("\n" + "=" * 70)
    print("GENERATING TEST VISUALIZATIONS")
    print("=" * 70)
    
    try:
        from visualization import (
            save_roc_curve,
            save_precision_recall_curve,
            save_confusion_matrix,
            save_model_comparison,
            save_feature_importance,
            save_class_distribution
        )
    except ImportError:
        print("⚠️  Warning: visualization module not found. Skipping visualization generation.")
        return
    
    import os
    
    # Create output directory
    full_output_dir = f"/Workspace/Users/pdadzie2@optumcloud.com/Diabetes Hospital Readmission Prediction/{output_dir}"
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Extract data for visualizations
    best_predictions = test_results['best_predictions']
    comparison_df = test_results['comparison_df']
    best_model_name = test_results['best_model_name']
    
    # Get numpy arrays for plotting
    y_true, y_prob, y_pred = get_roc_pr_data(best_predictions)
    
    # Get metrics from comparison DataFrame
    best_row = comparison_df.filter(col('Model') == best_model_name).first()
    auc_roc = best_row['AUC_ROC']
    auc_pr = best_row['AUC_PR']
    
    print(f"\n📊 Saving visualizations to: {full_output_dir}/")
    
    # 1. ROC Curve
    try:
        save_roc_curve(y_true, y_prob, auc_roc, f"{full_output_dir}/roc_curve.png", 
                      f"ROC Curve - {best_model_name}")
        print("   ✅ ROC curve saved")
    except Exception as e:
        print(f"   ❌ ROC curve failed: {e}")
    
    # 2. Precision-Recall Curve
    try:
        save_precision_recall_curve(y_true, y_prob, auc_pr, f"{full_output_dir}/precision_recall_curve.png",
                                   f"Precision-Recall Curve - {best_model_name}")
        print("   ✅ Precision-Recall curve saved")
    except Exception as e:
        print(f"   ❌ Precision-Recall curve failed: {e}")
    
    # 3. Confusion Matrix
    try:
        save_confusion_matrix(y_true, y_pred, f"{full_output_dir}/confusion_matrix.png",
                            ['No Readmit', 'Readmit'], f"Confusion Matrix - {best_model_name}")
        print("   ✅ Confusion matrix saved")
    except Exception as e:
        print(f"   ❌ Confusion matrix failed: {e}")
    
    # 4. Model Comparison
    try:
        comparison_pd = comparison_df.toPandas()
        save_model_comparison(comparison_pd, f"{full_output_dir}/model_comparison.png",
                            ['AUC_ROC', 'AUC_PR', 'F1_Score', 'Accuracy'], "Model Performance Comparison")
        print("   ✅ Model comparison saved")
    except Exception as e:
        print(f"   ❌ Model comparison failed: {e}")
    
    # 5. Class Distribution
    try:
        save_class_distribution(y_true, f"{full_output_dir}/class_distribution.png",
                              ['No Readmit', 'Readmit'], "Class Distribution in Test Set")
        print("   ✅ Class distribution saved")
    except Exception as e:
        print(f"   ❌ Class distribution failed: {e}")
    
    # 6. Feature Importance (if provided)
    if feature_names is not None and feature_importance is not None:
        try:
            save_feature_importance(feature_names, feature_importance, f"{full_output_dir}/feature_importance.png",
                                  15, f"Top 15 Feature Importances - {best_model_name}")
            print("   ✅ Feature importance saved")
        except Exception as e:
            print(f"   ❌ Feature importance failed: {e}")
    
    print("\n" + "=" * 70)
    print(f"✅ VISUALIZATION GENERATION COMPLETE")
    print(f"   All images saved to: {full_output_dir}/")
    print("=" * 70)


# ============================================================
# SECTION 5: FILE I/O FUNCTIONS
# ============================================================

def save_predictions(predictions: DataFrame, output_path: str) -> None:
    """
    Save model predictions to parquet.
    
    Args:
        predictions: DataFrame with predictions
        output_path: Path to save predictions
    """
    predictions.select(
        'encounter_id', 'patient_nbr', TARGET_COLUMN, 'prediction', 'probability'
    ).write.mode("overwrite").parquet(output_path)
    print(f"✅ Predictions saved to: {output_path}")


def save_comparison_results(comparison_df: DataFrame, output_path: str) -> None:
    """
    Save model comparison results.
    
    Args:
        comparison_df: DataFrame with comparison metrics
        output_path: Path to save results
    """
    comparison_df.write.mode("overwrite").parquet(output_path)
    print(f"✅ Comparison results saved to: {output_path}")


# ============================================================
# SECTION 6: MAIN PIPELINE FUNCTIONS
# ============================================================

def evaluate_and_compare(results: Dict[str, Tuple], 
                        spark: SparkSession,
                        selection_metric: str = 'AUC_ROC') -> Tuple[str, PipelineModel, DataFrame, DataFrame]:
    """
    Complete evaluation pipeline (simple version).
    
    Args:
        results: Dictionary mapping model names to (model, predictions)
        spark: SparkSession instance
        selection_metric: Metric to use for best model selection
    
    Returns:
        Tuple of (best_model_name, best_model, best_predictions, comparison_df)
    """
    # Compare all models
    comparison_df = compare_models(results, spark)
    
    # Select best model
    best_model_name, best_model, best_predictions = select_best_model(
        comparison_df, results, selection_metric
    )
    
    return best_model_name, best_model, best_predictions, comparison_df


def run_comprehensive_tests(results: Dict[str, Tuple], spark: SparkSession, 
                          train_df: DataFrame, test_df: DataFrame) -> Dict:
    """
    Run comprehensive testing suite on all models.
    
    Args:
        results: Dictionary of (model, predictions) tuples keyed by model name
        spark: SparkSession
        train_df: Training DataFrame
        test_df: Test DataFrame
    
    Returns:
        Dictionary with test results and best model info
    """
    print("=" * 70)
    print("COMPREHENSIVE MODEL TESTING")
    print("=" * 70)
    
    # Validate data leakage prevention
    validate_data_leakage_prevention(train_df, test_df)
    
    # Create evaluators
    evaluators = create_evaluators()
    
    # Compare all models
    comparison_df = compare_models(results, spark, evaluators)
    
    # Select best model
    best_model_name, best_model, best_predictions = select_best_model(
        comparison_df, results, 'AUC_ROC'
    )
    
    # Analyze confusion matrix
    cm, sensitivity, specificity = analyze_confusion_matrix(best_predictions, best_model_name)
    
    # Analyze prediction distribution
    predictions_detailed = analyze_prediction_distribution(best_predictions)
    
    # Test at different thresholds
    test_model_at_thresholds(best_predictions)
    
    # Prepare results
    test_results = {
        'best_model_name': best_model_name,
        'best_model': best_model,
        'best_predictions': best_predictions,
        'comparison_df': comparison_df,
        'confusion_matrix': cm,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'predictions_detailed': predictions_detailed
    }
    
    print("\n" + "=" * 70)
    print("✅ COMPREHENSIVE TESTING COMPLETE")
    print("=" * 70)
    
    return test_results
