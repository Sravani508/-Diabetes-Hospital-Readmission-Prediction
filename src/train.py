"""
Model Training Module for Hospital Readmission Prediction

This module handles:
- Train/test split by patient ID (prevents data leakage)
- Feature encoding and scaling
- Class imbalance handling
- Model training (Logistic Regression, Random Forest, GBT)
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, lit
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline, PipelineModel
from typing import Tuple, List, Dict


TARGET_COLUMN = "readmit_30"
PATIENT_ID_COLUMN = "patient_nbr"
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
RANDOM_SEED = 42


def prepare_train_test_split(df: DataFrame, train_ratio: float = TRAIN_RATIO,
                             test_ratio: float = TEST_RATIO,
                             seed: int = RANDOM_SEED) -> Tuple[DataFrame, DataFrame]:
    """
    Split data into train and test sets by patient ID.
    
    This ensures the same patient doesn't appear in both train and test sets,
    preventing data leakage when patients have multiple hospital visits.
    """
    # Get unique patient IDs
    unique_patients = df.select(PATIENT_ID_COLUMN).distinct()
    
    # Split patient IDs into train and test groups
    train_patients, test_patients = unique_patients.randomSplit([train_ratio, test_ratio], seed=seed)
    
    # Filter original dataframe based on patient groups
    train_df = df.join(train_patients, on=PATIENT_ID_COLUMN, how="inner")
    test_df = df.join(test_patients, on=PATIENT_ID_COLUMN, how="inner")
    
    # Print statistics
    train_count = train_df.count()
    test_count = test_df.count()
    train_patient_count = train_patients.count()
    test_patient_count = test_patients.count()
    
    print(f"Training: {train_count:,} visits from {train_patient_count:,} patients")
    print(f"Test: {test_count:,} visits from {test_patient_count:,} patients")
    print(f"Split ratio: {train_count/(train_count+test_count):.2%} train / {test_count/(train_count+test_count):.2%} test")
    
    return train_df, test_df


def calculate_class_weights(train_df: DataFrame) -> Tuple[float, float]:
    """Calculate class weights for imbalanced dataset."""
    class_counts = train_df.groupBy(TARGET_COLUMN).count().collect()
    class_dict = {row[TARGET_COLUMN]: row['count'] for row in class_counts}
    neg_count, pos_count = class_dict[0], class_dict[1]
    total_count = neg_count + pos_count
    weight_neg = total_count / (2 * neg_count)
    weight_pos = total_count / (2 * pos_count)
    print(f"Weights - Negative: {weight_neg:.4f}, Positive: {weight_pos:.4f}")
    return weight_neg, weight_pos


def add_sample_weights(train_df: DataFrame, weight_neg: float, weight_pos: float) -> DataFrame:
    """Add sample weights column."""
    return train_df.withColumn('sample_weight',
        when(col(TARGET_COLUMN) == 0, lit(weight_neg)).otherwise(lit(weight_pos)))


def create_feature_pipeline(numeric_features: List[str], binary_features: List[str],
                           categorical_features: List[str]) -> Tuple[Pipeline, str]:
    """Create feature encoding and scaling pipeline."""
    stages = []
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_indexed", handleInvalid="keep")
                for c in categorical_features]
    stages.extend(indexers)
    encoders = [OneHotEncoder(inputCol=f"{c}_indexed", outputCol=f"{c}_encoded")
                for c in categorical_features]
    stages.extend(encoders)
    encoded_categorical = [f"{c}_encoded" for c in categorical_features]
    all_features = numeric_features + binary_features + encoded_categorical
    assembler = VectorAssembler(inputCols=all_features, outputCol="features_unscaled", handleInvalid="skip")
    stages.append(assembler)
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withStd=True, withMean=False)
    stages.append(scaler)
    return Pipeline(stages=stages), "features"


def train_all_models(train_df: DataFrame, test_df: DataFrame, numeric_features: List[str],
                    binary_features: List[str], categorical_features: List[str]):
    """Complete training pipeline for all models."""
    weight_neg, weight_pos = calculate_class_weights(train_df)
    train_df_weighted = add_sample_weights(train_df, weight_neg, weight_pos)
    feature_pipeline, features_col = create_feature_pipeline(numeric_features, binary_features, categorical_features)
    feature_pipeline_model = feature_pipeline.fit(train_df_weighted)
    train_transformed = feature_pipeline_model.transform(train_df_weighted)
    test_transformed = feature_pipeline_model.transform(test_df)
    
    results = {}
    
    # Logistic Regression
    lr = LogisticRegression(featuresCol=features_col, labelCol=TARGET_COLUMN, weightCol="sample_weight",
                           maxIter=100, regParam=0.01, elasticNetParam=0.5)
    lr_model = lr.fit(train_transformed)
    results['Logistic Regression'] = (lr_model, lr_model.transform(test_transformed))
    print("✅ Logistic Regression trained")
    
    # Random Forest
    rf = RandomForestClassifier(featuresCol=features_col, labelCol=TARGET_COLUMN, weightCol="sample_weight",
                               numTrees=100, maxDepth=10, minInstancesPerNode=5, seed=RANDOM_SEED)
    rf_model = rf.fit(train_transformed)
    results['Random Forest'] = (rf_model, rf_model.transform(test_transformed))
    print("✅ Random Forest trained")
    
    # GBT
    gbt = GBTClassifier(featuresCol=features_col, labelCol=TARGET_COLUMN, weightCol="sample_weight",
                       maxIter=100, maxDepth=5, stepSize=0.1, seed=RANDOM_SEED)
    gbt_model = gbt.fit(train_transformed)
    results['Gradient Boosted Trees'] = (gbt_model, gbt_model.transform(test_transformed))
    print("✅ GBT trained")
    
    return results, feature_pipeline_model


def save_model(model: PipelineModel, output_path: str) -> None:
    """Save trained model."""
    model.write().overwrite().save(output_path)
    print(f"✅ Model saved to: {output_path}")
