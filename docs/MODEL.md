# Healthcare Insurance Risk Prediction Model

## Overview

This document provides comprehensive guidance for data scientists working with the healthcare insurance risk prediction model. The model predicts healthcare insurance costs or health risk scores for patients using demographic, health, and behavioral features while maintaining HIPAA compliance and healthcare industry standards.

## Model Architecture

### Target Variables
The model supports two prediction targets:
- **Insurance Charges** (`charges`): Dollar amounts for healthcare insurance costs
- **Health Risk Score** (`health_risk_score`): Normalized risk score (0-100 scale)

### Model Types
Two ensemble algorithms are supported:

#### 1. Random Forest Regressor
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

#### 2. Gradient Boosting Regressor  
```python
GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
```

## Feature Engineering

### Raw Features from Source Tables
The model uses the following base features from `dim_patients` table:

| Feature | Source Column | Description |
|---------|---------------|-------------|
| `patient_gender` | `patient_gender` | Gender (M/F) (categorical) |
| `patient_region` | `patient_region` | Geographic region (NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST) |
| `patient_age_category` | `patient_age_category` | Age group (YOUNG_ADULT, ADULT, MIDDLE_AGE, SENIOR) |
| `bmi` | `bmi` | Body Mass Index (numeric, already calculated) |
| `patient_smoking_status` | `patient_smoking_status` | Smoking status (SMOKER, NON_SMOKER) |
| `family_size_category` | `family_size_category` | Family size category (INDIVIDUAL, SMALL_FAMILY, etc.) |
| `number_of_dependents` | `number_of_dependents` | Number of dependents (numeric) |

### Engineered Features
The feature engineering pipeline creates the following derived features:

#### 1. Age Risk Score (`age_risk_score`)
```python
# Age risk scoring (1-5 scale)
age_risk_score = CASE 
    WHEN age < 25 THEN 1
    WHEN age < 35 THEN 2  
    WHEN age < 50 THEN 3
    WHEN age < 65 THEN 4
    ELSE 5 
END
```

#### 2. Smoking Impact Factor (`smoking_impact`)
```python
# Smoking amplifies age-related risk
smoking_impact = CASE 
    WHEN smoker THEN age * 2.5 
    ELSE age * 1.0 
END
```

#### 3. Family Size Factor (`family_size_factor`)
```python
# Family size adjustment (15% per child)
family_size_factor = 1 + (children * 0.15)
```

#### 4. Regional Multiplier (`regional_multiplier`)
```python
# Regional cost adjustments
regional_multiplier = CASE 
    WHEN region = 'NORTHEAST' THEN 1.2
    WHEN region = 'NORTHWEST' THEN 1.1  
    WHEN region = 'SOUTHEAST' THEN 1.0
    ELSE 0.95 
END
```

#### 5. Health Risk Composite (`health_risk_composite`)
```python
# Composite health risk calculation
health_risk_composite = (age_risk_score * 20) + 
                       (CASE WHEN smoker THEN 50 ELSE 0 END) + 
                       (CASE WHEN bmi > 30 THEN 30 ELSE 0 END)
```

#### 6. Data Quality Score (`data_quality_score`)
- Derived from `patient_data_quality_score` in source table
- Used for model confidence assessment

### Feature Preprocessing Pipeline

#### Categorical Encoding
- **Method**: LabelEncoder for categorical features
- **Features**: `sex`, `region` (from feature table)
- **Encoding**: Stored with the model for consistent inference
- **Note**: Feature table aliases `patient_gender` → `sex` and keeps `patient_region` → `region`

#### Numerical Scaling
- **Method**: StandardScaler applied to all features
- **Features**: All numerical features are z-score normalized
- **Preservation**: Scaler fitted on training data and stored with model

#### Feature Table Schema
The feature engineering process creates a feature table (`ml_insurance_features`) that includes:
- All columns from `dim_patients` (for reference)
- Engineered features (`age_risk_score`, `smoking_impact`, etc.)
- Mapped features (`age`, `children`, `smoker`, `sex`, `region`)
- A `customer_id` column (mapped from `patient_natural_key`) used as the primary key

**Key Mappings in Feature Table:**
```python
# Age category mapping (done in feature engineering)
age = CASE
    WHEN patient_age_category = 'YOUNG_ADULT' THEN 25
    WHEN patient_age_category = 'ADULT' THEN 35
    WHEN patient_age_category = 'MIDDLE_AGE' THEN 45
    WHEN patient_age_category = 'SENIOR' THEN 60
    ELSE 70
END

# Children count mapping (from number_of_dependents)
children = number_of_dependents

# Smoker boolean mapping
smoker = (patient_smoking_status = 'SMOKER')

# Gender alias
sex = patient_gender

# Region alias
region = patient_region
```

## Model Training Process

### Training Pipeline
1. **Data Loading**: Load from `dim_patients` with `is_current_record = True`
2. **Feature Engineering**: Create features using Databricks Feature Engineering Client
3. **Feature Store**: Store engineered features in `juan_dev.healthcare_data.ml_insurance_features`
4. **Training Set Creation**: Join base data with features using `FeatureLookup`
5. **Preprocessing**: Apply categorical encoding and numerical scaling
6. **Model Training**: Train Random Forest or Gradient Boosting model
7. **Evaluation**: Calculate performance metrics and healthcare-specific assessments
8. **Model Registration**: Log model with MLflow and Feature Engineering integration

### Feature Store Integration
```python
# Define feature lookups (simplified approach using feature_names)
feature_lookups = [
    FeatureLookup(
        table_name="juan_dev.healthcare_data.ml_insurance_features",
        lookup_key="customer_id",
        feature_names=["age_risk_score", "smoking_impact", "family_size_factor",
                      "regional_multiplier", "health_risk_composite", "data_quality_score",
                      "sex", "region"]  # All features fetched in single lookup
    )
]

# Create training set with automatic feature joining
training_set = fe.create_training_set(
    df=base_df.withColumn("customer_id", col("patient_natural_key")),
    feature_lookups=feature_lookups,
    label="health_risk_score",
    exclude_columns=["timestamp", "ingestion_timestamp", "effective_from_date", "effective_to_date",
                   "effective_date", "expiry_date", "patient_surrogate_key", "patient_id_hash",
                   "patient_name_hash", "_pipeline_env", "dimension_last_updated"]
)
```

**Note**: The feature table contains ALL columns from `dim_patients` plus the engineered features. The Feature Engineering Client automatically joins all needed features based on the `customer_id` key.

### Custom Pipeline Architecture
The model uses a custom pipeline class (`HealthcareRiskPipeline`) that encapsulates:
- Label encoders for categorical features
- StandardScaler for numerical features  
- Trained model (Random Forest or Gradient Boosting)
- Feature column definitions
- Preprocessing logic for consistent inference

## Model Evaluation

### Primary Metrics

#### Regression Metrics
- **R² Score**: Coefficient of determination (target: ≥ 0.70)
- **Mean Absolute Error (MAE)**: Average absolute prediction error (target: ≤ 15.0 for risk scores)
- **Root Mean Squared Error (RMSE)**: Square root of mean squared error

#### Cross-Validation
- **Method**: 5-fold cross-validation
- **Scoring**: R² score used for model selection
- **Purpose**: Assess model generalization and stability

#### Healthcare-Specific Metrics
- **High-Risk Accuracy**: Ability to identify high-risk patients (target: ≥ 60%)
- **High-Risk Threshold**: 95th percentile of target variable
- **Business Impact**: Evaluated on financial accuracy and clinical relevance

### Evaluation Code Example
```python
# Calculate primary metrics
r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)  
rmse = mean_squared_error(y_test, y_test_pred, squared=False)

# Cross-validation assessment
cv_scores = cross_val_score(base_model, X_scaled, y, cv=5, scoring='r2')

# Healthcare-specific evaluation
high_risk_threshold = y.quantile(0.95)
high_risk_accuracy = _evaluate_high_risk_predictions(y_test, y_test_pred, high_risk_threshold)
```

## Model Governance & Promotion

### Governance Requirements
Models must pass these healthcare industry standards before promotion:

| Metric | Minimum Threshold | Purpose |
|--------|------------------|---------|
| R² Score | ≥ 0.70 | Overall predictive accuracy |
| MAE | ≤ 15.0 | Acceptable prediction error (risk scores) |
| High-Risk Accuracy | ≥ 60% | Critical patient identification |

### Promotion Process
1. **Validation**: Check metrics against healthcare requirements
2. **Compliance Tagging**: Add `healthcare_compliance: validated` tag
3. **Metadata Update**: Update model description with performance metrics and compliance status
4. **Alias Assignment**: Set `champion` alias for production deployment
5. **Governance Logging**: Record governance decision and rationale

### Champion/Challenger Pattern
Optional governance pattern for A/B testing new models:
- **Champion**: Current production model with `champion` alias
- **Challenger**: New candidate model for evaluation
- **Promotion Criteria**: Challenger must outperform champion on primary metrics without significant degradation on secondary metrics

### Governance Code Example
```python
class ModelGovernance:
    def _validate_healthcare_requirements(self, metrics):
        requirements = {
            "min_r2_score": 0.70,
            "max_mae": 15.0, 
            "min_high_risk_accuracy": 0.60
        }
        
        r2_check = metrics.get('r2_score', 0) >= requirements['min_r2_score']
        mae_check = metrics.get('mean_absolute_error', float('inf')) <= requirements['max_mae']
        accuracy_check = metrics.get('high_risk_accuracy', 0) >= requirements['min_high_risk_accuracy']
        
        return r2_check and mae_check and accuracy_check
```

## Model Deployment & Inference

### Batch Inference
Models are deployed for batch scoring using the Feature Engineering Client:

```python
# Load model from Unity Catalog
model_uri = f"models:/{model_name}@champion"

# Prepare input data (only need to create customer_id mapping)
input_df_prepared = (
    input_df
    .withColumn("customer_id", col("patient_natural_key"))
)

# Batch scoring with automatic feature lookup
# The Feature Engineering Client automatically:
# 1. Joins with feature table using customer_id
# 2. Fetches all required features
# 3. Applies model preprocessing pipeline
# 4. Generates predictions
predictions_df = fe.score_batch(
    df=input_df_prepared,
    model_uri=model_uri
)
```

**Important**: Unlike training, batch inference does NOT require manual feature engineering. The Feature Engineering Client handles all feature lookup and preprocessing automatically using the model's embedded feature metadata.

### Business Logic Post-Processing
After model prediction, business rules are applied:

#### Risk Categorization
```python
risk_category = CASE 
    WHEN adjusted_prediction < 30 THEN 'low'
    WHEN adjusted_prediction < 60 THEN 'medium' 
    WHEN adjusted_prediction < 85 THEN 'high'
    ELSE 'critical' 
END
```

#### Business Rules
- **Minimum Risk Score**: 10 (no prediction below this threshold)
- **High-Risk Flag**: `adjusted_prediction > 75 OR risk_category = 'critical'`
- **Review Flag**: `adjusted_prediction > 90`
- **Confidence Intervals**: ±10% of prediction for business planning

#### Output Table Schema
Batch inference results are saved to `juan_dev.healthcare_data.ml_patient_predictions` with the following additional columns:
- `prediction`: Raw model output
- `adjusted_prediction`: Business-adjusted prediction (minimum 10)
- `risk_category`: low/medium/high/critical
- `high_risk_patient`: Boolean flag
- `requires_review`: Boolean flag
- `prediction_lower_bound`: Lower confidence bound
- `prediction_upper_bound`: Upper confidence bound
- `prediction_timestamp`: When prediction was generated
- `model_version`: Model version used
- `model_name`: Full model name from Unity Catalog

## Model Monitoring

### Drift Detection
The model includes comprehensive monitoring for:
- **Statistical Drift**: Changes in prediction distributions
- **Demographic Drift**: Shifts in patient population characteristics  
- **Performance Drift**: Degradation in model accuracy when ground truth available
- **Regional Equity**: Monitoring for geographic bias

### Key Monitoring Metrics
- **Prediction Drift**: >20% change in average predictions triggers alert
- **Volume Monitoring**: <50 daily predictions triggers pipeline health check
- **Accuracy Monitoring**: MAE >$2,500 (cost model) or >15 (risk model) triggers retraining alert
- **Demographic Fairness**: >10% shift in demographic distributions triggers bias review

### Monitoring Views
- `juan_dev.ml.healthcare_drift_detection`: Daily drift analysis
- `juan_dev.ml.healthcare_model_alerts`: Active performance alerts
- `juan_dev.ml.model_performance_dashboard`: Executive summary

## Healthcare Compliance

### HIPAA Compliance
- **Data Processing**: All data handling follows HIPAA deidentification standards
- **Model Tags**: Models tagged with `hipaa_compliant` status
- **Audit Trail**: Complete lineage tracking through Unity Catalog
- **Access Controls**: Role-based access to sensitive model components

### Clinical Validation
- **BMI Standards**: Clinical BMI categorization thresholds
- **Age Risk**: Actuarial age-based risk scoring aligned with industry standards
- **Smoking Impact**: Evidence-based smoking risk multipliers
- **Bias Testing**: Regular evaluation for demographic and geographic bias

### Business Impact Validation
- **Cost Accuracy**: Models validated against actual claims data when available
- **High-Risk Identification**: Validated ability to identify patients requiring intervention
- **Pricing Fairness**: Regional and demographic equity assessments
- **Population Health**: Models contribute to broader healthcare analytics and trends

## Getting Started for New DS Team Members

### Prerequisites
1. Access to Databricks workspace with Unity Catalog
2. Permissions for `juan_dev.healthcare_data` schema
3. MLflow access with Unity Catalog integration
4. Understanding of healthcare compliance requirements

### Quick Start Workflow
1. **Verify Data**: Ensure prerequisite tables exist and have recent data
2. **Feature Engineering**: Run `00-training/00-insurance-model-feature.ipynb`
3. **Model Training**: Execute `00-training/01-insurance-model-train.ipynb`
4. **Governance Check**: Run `01-governance/insurance-model-governance.py`
5. **Batch Inference**: Test with `02-batch/insurance-model-batch.ipynb`
6. **Setup Monitoring**: Initialize with `03-monitoring/insurance-model-monitor.ipynb`

### Key Concepts to Understand
- **Feature Store**: Centralized feature management with automatic lookup
- **Unity Catalog**: Governance and lineage for all ML assets
- **Healthcare Compliance**: Industry-specific validation requirements
- **Custom Pipeline**: Embedded preprocessing for consistent inference
- **Drift Monitoring**: Proactive detection of model degradation and bias

### Common Debugging Steps
1. **Data Issues**: Verify prerequisite tables exist with `SELECT * FROM juan_dev.healthcare_data.dim_patients LIMIT 10`
2. **Feature Issues**: Check feature store with `SELECT * FROM juan_dev.healthcare_data.ml_insurance_features LIMIT 10`
3. **Model Issues**: Validate champion model exists with MLflow client
4. **Governance Issues**: Review validation thresholds and metrics calculation
5. **Monitoring Issues**: Run diagnostics with `monitor.diagnose_and_fix_setup_issues()`

### Important Schema Notes
**Key Column Name Differences to Remember:**
- `dim_patients.patient_gender` → Feature table: `sex`
- `dim_patients.patient_region` → Feature table: `region`
- `dim_patients.bmi` → Already numeric (not a category)
- `dim_patients.number_of_dependents` → Feature table: `children`
- `dim_patients.patient_natural_key` → Feature table: `customer_id` (primary key)

**Feature Table Contents:**
The feature table (`ml_insurance_features`) contains:
- **All original columns** from `dim_patients` (for reference and debugging)
- **Engineered features** (`age_risk_score`, `smoking_impact`, `family_size_factor`, `regional_multiplier`, `health_risk_composite`, `data_quality_score`)
- **Mapped features** (`age`, `sex`, `region`, `bmi`, `children`, `smoker`) used by the model
- **Primary key** `customer_id` for joining

**Model Feature Expectations:**
The trained model expects exactly these 8 features:
1. `age_risk_score` (int)
2. `smoking_impact` (decimal)
3. `family_size_factor` (decimal)
4. `health_risk_composite` (double)
5. `regional_multiplier` (decimal)
6. `data_quality_score` (double)
7. `sex` (string → LabelEncoded)
8. `region` (string → LabelEncoded)

### Completed MLOps Pipeline
As of the last successful run, the complete pipeline has been executed:

✅ **Feature Engineering** (`feature_engineering_job`)
- Created feature table with 10,000 records
- Engineered 6 numerical features + 2 categorical features
- Established `customer_id` as primary key

✅ **Model Training** (`model_training_job`)
- Trained Random Forest Regressor (Version 1)
- Target: `health_risk_score` (0-100 scale)
- Registered in Unity Catalog: `juan_dev.healthcare_data.insurance_model`

✅ **Model Governance** (`model_governance_job`)
- Validated against healthcare requirements
- Promoted model with "champion" alias
- Tagged with compliance metadata

✅ **Batch Inference** (`batch_inference_job`)
- Processed 10,000 patient records
- Generated risk scores and categories
- Saved to `juan_dev.healthcare_data.ml_patient_predictions`
- Distribution: 33% low, 13% medium, 45% high, 9% critical risk

This model represents a production-ready healthcare ML system with enterprise governance, compliance, and monitoring capabilities designed for the healthcare insurance industry.