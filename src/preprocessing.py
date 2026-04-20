"""
Data Preprocessing Module for Hospital Readmission Prediction

This module handles:
- Data loading and validation
- Missing value treatment
- Invalid record removal (expired/hospice patients)
- Feature engineering
- Data type conversions
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, lit, count, avg, sum as spark_sum,
    create_map, round as spark_round
)
from pyspark.sql.types import DoubleType
from itertools import chain
from typing import List, Dict, Tuple


# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

# Missing value settings
MISSING_INDICATOR = "?"
MISSING_THRESHOLD = 50.0  # Drop columns with more than 50% missing

# Reference mappings
ADMISSION_TYPE_MAP = {
    1: 'Emergency', 2: 'Urgent', 3: 'Elective', 4: 'Newborn',
    5: 'Not Available', 6: 'NULL', 7: 'Trauma Center', 8: 'Not Mapped'
}

DISCHARGE_MAP = {
    1: 'Discharged to home', 2: 'Transferred to short term hospital',
    3: 'Transferred to SNF', 4: 'Transferred to ICF',
    5: 'Transferred to other inpatient care', 6: 'Home with home health service',
    7: 'Left AMA', 8: 'Home under care of Home IV provider',
    9: 'Admitted as inpatient to this hospital', 10: 'Neonate transferred for aftercare',
    11: 'Expired', 12: 'Still patient/expected to return',
    13: 'Hospice / home', 14: 'Hospice / medical facility',
    15: 'Transferred to Medicare swing bed', 16: 'Transferred for outpatient services',
    17: 'Referred to this institution outpatient', 18: 'NULL',
    19: 'Expired at home (Medicaid hospice)', 20: 'Expired in medical facility (Medicaid hospice)',
    21: 'Expired place unknown (Medicaid hospice)', 22: 'Transferred to rehab facility',
    23: 'Transferred to long term care hospital', 24: 'Transferred to Medicaid nursing facility',
    25: 'Not Mapped', 26: 'Unknown/Invalid', 27: 'Transferred to federal health care facility',
    28: 'Transferred to psychiatric hospital', 29: 'Transferred to Critical Access Hospital',
    30: 'Transferred to other health care institution'
}

ADMISSION_SOURCE_MAP = {
    1: 'Physician Referral', 2: 'Clinic Referral', 3: 'HMO Referral',
    4: 'Transfer from hospital', 5: 'Transfer from SNF',
    6: 'Transfer from other health care facility', 7: 'Emergency Room',
    8: 'Court/Law Enforcement', 9: 'Not Available',
    10: 'Transfer from Critical Access Hospital', 11: 'Normal Delivery',
    12: 'Premature Delivery', 13: 'Sick Baby', 14: 'Extramural Birth',
    15: 'Not Available', 17: 'NULL', 18: 'Transfer from Home Health Agency',
    19: 'Readmission to Same Home Health Agency', 20: 'Not Mapped',
    21: 'Unknown/Invalid', 22: 'Transfer from hospital inpatient (separate claim)',
    23: 'Born inside this hospital', 24: 'Born outside this hospital',
    25: 'Transfer from Ambulatory Surgery Center', 26: 'Transfer from Hospice'
}

# Patients who died or went to hospice - cannot be readmitted
EXCLUDE_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]


# ============================================================
# DATA LOADING
# ============================================================

def load_data(spark: SparkSession, file_path: str) -> DataFrame:
    """
    Load CSV data from specified path.
    
    Args:
        spark: SparkSession instance
        file_path: Full path to CSV file
        
    Returns:
        DataFrame with raw data
    """
    print(f"Loading data from: {file_path}")
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    total_rows = df.count()
    total_cols = len(df.columns)
    print(f"✅ Data loaded: {total_rows:,} rows, {total_cols} columns")
    
    return df


# ============================================================
# MISSING VALUE HANDLING
# ============================================================

def replace_missing_indicators(df: DataFrame, indicator: str = MISSING_INDICATOR) -> DataFrame:
    """
    Replace missing value indicators (e.g., '?') with NULL.
    
    Args:
        df: Input DataFrame
        indicator: String representing missing values
        
    Returns:
        DataFrame with proper NULL values
    """
    for column in df.columns:
        df = df.withColumn(
            column,
            when(col(column) == indicator, None).otherwise(col(column))
        )
    
    print(f"✅ Replaced '{indicator}' with NULL values")
    return df


def identify_high_missing_columns(df: DataFrame, threshold: float = MISSING_THRESHOLD) -> List[str]:
    """
    Identify columns with missing values above threshold.
    
    Args:
        df: Input DataFrame
        threshold: Percentage threshold for dropping columns
        
    Returns:
        List of column names to drop
    """
    total_rows = df.count()
    high_missing_cols = []
    
    for c in df.columns:
        null_count = df.filter(col(c).isNull()).count()
        missing_pct = (null_count / total_rows) * 100
        
        if missing_pct > threshold:
            high_missing_cols.append(c)
            print(f"   - {c}: {missing_pct:.2f}% missing")
    
    return high_missing_cols


def fill_missing_values(df: DataFrame) -> DataFrame:
    """
    Fill remaining missing values with appropriate defaults.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values filled
    """
    # Categorical columns - fill with 'Unknown'
    categorical_cols = ['race', 'gender', 'medical_specialty', 'payer_code',
                       'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed']
    
    for c in categorical_cols:
        if c in df.columns:
            null_count = df.filter(col(c).isNull()).count()
            if null_count > 0:
                df = df.withColumn(
                    c,
                    when(col(c).isNull(), 'Unknown').otherwise(col(c))
                )
    
    # Numeric columns - fill with median
    numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                   'num_medications', 'number_outpatient', 'number_emergency',
                   'number_inpatient', 'number_diagnoses']
    
    for c in numeric_cols:
        if c in df.columns:
            null_count = df.filter(col(c).isNull()).count()
            if null_count > 0:
                median_val = df.approxQuantile(c, [0.5], 0.01)[0]
                df = df.withColumn(
                    c,
                    when(col(c).isNull(), lit(median_val)).otherwise(col(c))
                )
    
    print("✅ Missing values filled")
    return df


# ============================================================
# DATA CLEANING
# ============================================================

def create_target_variable(df: DataFrame) -> DataFrame:
    """
    Create binary target variable for 30-day readmission.
    
    Args:
        df: Input DataFrame with 'readmitted' column
        
    Returns:
        DataFrame with 'readmit_30' binary target
    """
    df = df.withColumn(
        'readmit_30',
        when(col('readmitted') == '<30', 1).otherwise(0)
    )
    
    print("✅ Created binary target: readmit_30")
    return df


def remove_invalid_records(df: DataFrame, exclude_ids: List[int] = EXCLUDE_DISCHARGE_IDS) -> DataFrame:
    """
    Remove records for patients who died or went to hospice.
    
    Args:
        df: Input DataFrame
        exclude_ids: List of discharge disposition IDs to exclude
        
    Returns:
        Filtered DataFrame
    """
    count_before = df.count()
    df_valid = df.filter(~col('discharge_disposition_id').isin(exclude_ids))
    count_after = df_valid.count()
    removed = count_before - count_after
    
    print(f"✅ Removed {removed:,} invalid records ({removed/count_before*100:.2f}%)")
    return df_valid


def clean_gender_column(df: DataFrame) -> DataFrame:
    """
    Standardize gender values to Male/Female/Unknown.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned gender column
    """
    df = df.withColumn(
        'gender',
        when(col('gender').isin(['Male', 'Female']), col('gender'))
        .otherwise('Unknown')
    )
    
    print("✅ Gender column cleaned")
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_admission_features(df: DataFrame) -> DataFrame:
    """
    Create admission-related features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new admission features
    """
    # Admission type group
    df = df.withColumn(
        'admission_type_group',
        when(col('admission_type_id') == 1, 'Emergency')
        .when(col('admission_type_id') == 2, 'Urgent')
        .when(col('admission_type_id') == 3, 'Elective')
        .when(col('admission_type_id') == 4, 'Newborn')
        .otherwise('Other_Unknown')
    )
    
    # Emergency admission flag
    df = df.withColumn(
        'emergency_admission',
        when(col('admission_type_id') == 1, 1).otherwise(0)
    )
    
    # Admission source group
    df = df.withColumn(
        'admission_source_group',
        when(col('admission_source_id') == 7, 'Emergency_Room')
        .when(col('admission_source_id') == 1, 'Physician_Referral')
        .when(col('admission_source_id').isin([2, 3]), 'Clinic_HMO_Referral')
        .when(col('admission_source_id').isin([4, 5, 6, 10, 22, 25, 26]), 'Transfer_Facility')
        .otherwise('Other')
    )
    
    # Admitted from ER flag
    df = df.withColumn(
        'admitted_from_er',
        when(col('admission_source_id') == 7, 1).otherwise(0)
    )
    
    print("✅ Created admission features")
    return df


def create_discharge_features(df: DataFrame) -> DataFrame:
    """
    Create discharge-related features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new discharge features
    """
    # Discharge group
    df = df.withColumn(
        'discharge_group',
        when(col('discharge_disposition_id').isin([1, 6, 8]), 'Home')
        .when(col('discharge_disposition_id').isin([2, 3, 4, 5, 15, 22, 23, 24, 27, 28, 29, 30]), 'Transfer_Facility')
        .when(col('discharge_disposition_id') == 7, 'Left_AMA')
        .otherwise('Other')
    )
    
    # Discharged not to home flag
    df = df.withColumn(
        'discharged_not_home',
        when(col('discharge_disposition_id').isin([1, 6, 8]), 0).otherwise(1)
    )
    
    # Left AMA flag
    df = df.withColumn(
        'left_ama',
        when(col('discharge_disposition_id') == 7, 1).otherwise(0)
    )
    
    print("✅ Created discharge features")
    return df


def create_utilization_features(df: DataFrame) -> DataFrame:
    """
    Create healthcare utilization features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with utilization features
    """
    # Total prior visits
    df = df.withColumn(
        'total_prior_visits',
        col('number_outpatient') + col('number_emergency') + col('number_inpatient')
    )
    
    # High utilizer flag (>= 3 prior visits)
    df = df.withColumn(
        'high_utilizer',
        when(col('total_prior_visits') >= 3, 1).otherwise(0)
    )
    
    # Frequent ER visitor (>= 2 ER visits)
    df = df.withColumn(
        'frequent_er_visitor',
        when(col('number_emergency') >= 2, 1).otherwise(0)
    )
    
    # Complexity score
    df = df.withColumn(
        'complexity_score',
        (col('num_procedures') + col('num_medications') + col('number_diagnoses')) / 3.0
    )
    
    print("✅ Created utilization features")
    return df


def create_age_features(df: DataFrame) -> DataFrame:
    """
    Create age-related features.
    
    Args:
        df: Input DataFrame with age ranges
        
    Returns:
        DataFrame with numeric age and elderly flag
    """
    # Age mapping to numeric midpoint
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    
    age_expr = create_map([lit(x) for x in chain(*age_map.items())])
    
    df = df.withColumn('age_numeric', age_expr[col('age')])
    
    # Elderly flag (>= 65)
    df = df.withColumn(
        'is_elderly',
        when(col('age_numeric') >= 65, 1).otherwise(0)
    )
    
    print("✅ Created age features")
    return df


def create_lab_test_features(df: DataFrame) -> DataFrame:
    """
    Create lab test and medication features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with lab/medication features
    """
    # A1C tested
    df = df.withColumn(
        'a1c_tested',
        when(col('A1Cresult') != 'None', 1).otherwise(0)
    )
    
    # A1C abnormal
    df = df.withColumn(
        'a1c_abnormal',
        when(col('A1Cresult').isin(['>7', '>8']), 1).otherwise(0)
    )
    
    # Glucose tested
    df = df.withColumn(
        'glucose_tested',
        when(col('max_glu_serum') != 'None', 1).otherwise(0)
    )
    
    # On insulin
    df = df.withColumn(
        'on_insulin',
        when(col('insulin').isin(['Down', 'Steady', 'Up']), 1).otherwise(0)
    )
    
    # Medication changed
    df = df.withColumn(
        'med_changed',
        when(col('change') == 'Ch', 1).otherwise(0)
    )
    
    print("✅ Created lab test features")
    return df


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def preprocess_data(spark: SparkSession, file_path: str) -> Tuple[DataFrame, List[str], List[str], List[str]]:
    """
    Complete preprocessing pipeline.
    
    Args:
        spark: SparkSession instance
        file_path: Path to raw data CSV
        
    Returns:
        Tuple of (processed_df, numeric_features, binary_features, categorical_features)
    """
    print("=" * 60)
    print("STARTING DATA PREPROCESSING")
    print("=" * 60)
    
    # Load data
    df = load_data(spark, file_path)
    
    # Create target variable
    df = create_target_variable(df)
    
    # Handle missing values
    df = replace_missing_indicators(df)
    
    # Remove invalid records
    df = remove_invalid_records(df)
    
    # Identify and drop high-missing columns
    high_missing_cols = identify_high_missing_columns(df)
    if high_missing_cols:
        print(f"Dropping {len(high_missing_cols)} high-missing columns")
        df = df.drop(*high_missing_cols)
    
    # Fill remaining missing values
    df = fill_missing_values(df)
    
    # Clean categorical columns
    df = clean_gender_column(df)
    
    # Feature engineering
    df = create_admission_features(df)
    df = create_discharge_features(df)
    df = create_utilization_features(df)
    df = create_age_features(df)
    df = create_lab_test_features(df)
    
    # Define feature lists
    numeric_features = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses', 'total_prior_visits',
        'complexity_score', 'age_numeric'
    ]
    
    binary_features = [
        'emergency_admission', 'admitted_from_er', 'discharged_not_home',
        'left_ama', 'high_utilizer', 'frequent_er_visitor', 'is_elderly',
        'a1c_tested', 'a1c_abnormal', 'glucose_tested', 'on_insulin', 'med_changed'
    ]
    
    categorical_features = [
        'race', 'gender', 'admission_type_group',
        'admission_source_group', 'discharge_group'
    ]
    
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print(f"Final dataset: {df.count():,} rows, {len(df.columns)} columns")
    print(f"Features: {len(numeric_features)} numeric, {len(binary_features)} binary, {len(categorical_features)} categorical")
    print("=" * 60)
    
    return df, numeric_features, binary_features, categorical_features
