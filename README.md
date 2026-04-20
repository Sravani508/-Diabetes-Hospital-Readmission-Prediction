# Data Directory

## Dataset Information

This project uses the **Diabetes 130-US Hospitals for Years 1999-2008** dataset.

### Source

* **Repository**: UCI Machine Learning Repository
* **URL**: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008
* **Citation**: Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records," BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.

### Dataset Description

The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes.

**Key Statistics:**
* **Records**: 101,766 hospital encounters
* **Patients**: ~70,000 unique patients
* **Hospitals**: 130 US hospitals
* **Time Period**: 1999-2008
* **Features**: 50+ attributes

### Data Files

The raw data file is **NOT** included in this repository due to size and licensing considerations.

**To use this project:**

1. Download the dataset from the UCI ML Repository (link above)
2. Extract the `diabetic_data.csv` file
3. Place it in your data storage location
4. Update the file path in your configuration:
   ```python
   DATA_CONTAINER = "your-container"
   DATA_FOLDER = "your-folder"
   DATA_FILE = "diabetic_data.csv"
   ```

### Data Schema

#### Patient Demographics
* `race`: Caucasian, Asian, African American, Hispanic, Other
* `gender`: Male, Female, Unknown/Invalid
* `age`: Grouped in 10-year intervals [0-10), [10-20), ..., [90-100)

#### Admission Details
* `admission_type_id`: Integer identifier for admission type
* `admission_source_id`: Integer identifier for admission source
* `time_in_hospital`: Integer number of days (1-14)

#### Discharge Information
* `discharge_disposition_id`: Integer identifier for discharge disposition
* `readmitted`: Target variable - "<30" (readmitted within 30 days), ">30" (readmitted after 30 days), "NO" (not readmitted)

#### Clinical Information
* `num_lab_procedures`: Number of lab tests performed
* `num_procedures`: Number of procedures (other than lab tests)
* `num_medications`: Number of distinct generic medications
* `number_outpatient`: Number of outpatient visits in the year before encounter
* `number_emergency`: Number of emergency visits in the year before encounter
* `number_inpatient`: Number of inpatient visits in the year before encounter
* `number_diagnoses`: Number of diagnoses entered (up to 9)

#### Diagnosis Codes
* `diag_1`, `diag_2`, `diag_3`: Primary, secondary, and tertiary diagnoses (ICD-9 codes)

#### Lab Results
* `max_glu_serum`: Glucose serum test result (">200", ">300", "normal", "none")
* `A1Cresult`: HbA1c test result (">7", ">8", "normal", "none")

#### Medications (24 features)
Indicators for various diabetes medications including:
* `metformin`, `insulin`, `glipizide`, `glyburide`, etc.
* Values: "No" (not prescribed), "Steady" (no change), "Up" (dosage increased), "Down" (dosage decreased)

#### Other Features
* `change`: Indicates if there was a change in diabetic medications ("Ch" or "No")
* `diabetesMed`: Indicates if any diabetic medication was prescribed ("Yes" or "No")
* `encounter_id`: Unique identifier for each encounter
* `patient_nbr`: Unique identifier for each patient

### Missing Values

The dataset uses `?` to represent missing values. The preprocessing pipeline handles this by:
1. Converting `?` to NULL
2. Dropping columns with >50% missing values
3. Filling remaining missing values with appropriate defaults

**Columns with High Missing Rates:**
* `weight`: ~97% missing (dropped)
* `payer_code`: ~40% missing (filled with "Unknown")
* `medical_specialty`: ~49% missing (filled with "Unknown")

### Data Quality Notes

1. **Excluded Records**: Patients who died or went to hospice are excluded from analysis (cannot be readmitted)
2. **Duplicate Patients**: Some patients have multiple encounters; each encounter is treated independently
3. **ICD-9 Codes**: Diagnosis codes are in ICD-9 format (pre-ICD-10 era)
4. **Medication Names**: Generic medication names are used

### Target Variable

The original `readmitted` column has three values:
* `<30`: Readmitted within 30 days
* `>30`: Readmitted after 30 days
* `NO`: Not readmitted

**For this project**, we create a binary target variable `readmit_30`:
* `1`: Readmitted within 30 days (`<30`)
* `0`: Not readmitted within 30 days (`>30` or `NO`)

**Class Distribution:**
* Positive class (readmitted <30 days): ~11%
* Negative class (not readmitted <30 days): ~89%
* **Imbalance ratio**: ~8:1

### Data Privacy

This dataset has been de-identified and contains no protected health information (PHI). All patient and hospital identifiers have been removed or anonymized.

### Usage Guidelines

1. **Academic/Research Use**: Freely available for research purposes
2. **Citation Required**: Please cite the original paper when using this dataset
3. **No PHI**: Dataset is de-identified and HIPAA-compliant
4. **Commercial Use**: Check UCI ML Repository terms for commercial applications

### Preprocessing Pipeline

The `src/preprocessing.py` module handles:
* Loading CSV data
* Missing value treatment
* Invalid record removal
* Feature engineering
* Data type conversions

See the main README for usage examples.

---


