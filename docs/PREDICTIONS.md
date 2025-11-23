# Healthcare Insurance Prediction Flow Guide

## Overview

This guide explains how the healthcare insurance risk prediction system transforms raw patient data into actionable risk predictions. The system processes patient information through multiple stages—from initial data ingestion to final risk scores that inform care management and premium pricing decisions.

**Purpose**: Help technical users (data engineers, analysts, ML engineers) understand:
- How data flows through the prediction pipeline
- What transformations occur at each stage
- How to interpret prediction outputs
- What sample data looks like at each step

**Audience**: Technical users familiar with data pipelines, SQL, and machine learning concepts.

**Quick Links**:
- [MODEL.md](MODEL.md) - Complete model architecture and training documentation
- [TABLES.md](TABLES.md) - Detailed table schemas and data dictionary
- [MODEL-MONITORING.md](MODEL-MONITORING.md) - Monitoring infrastructure and metrics

---

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PREDICTION PIPELINE                             │
└─────────────────────────────────────────────────────────────────────┘

STAGE 1: RAW DATA INGESTION
┌──────────────────────────┐
│   bronze_patients        │  40,000 rows (raw streaming data)
│   - Demographics         │
│   - Health metrics       │
│   - Insurance info       │
└──────────┬───────────────┘
           │
           ↓ (Cleansing & validation)
           
STAGE 2: DATA TRANSFORMATION
┌──────────────────────────┐
│   silver_patients        │  40,000 rows (cleaned)
│   dim_patients           │  40,000 rows (dimensional model)
│   - SCD Type 2           │
│   - Enriched attributes  │
└──────────┬───────────────┘
           │
           ↓ (Feature engineering)

STAGE 3: FEATURE ENGINEERING
┌──────────────────────────┐
│ ml_insurance_features    │  10,000 rows (distinct customers)
│   - age_risk_score       │
│   - smoking_impact       │
│   - family_size_factor   │
│   - regional_multiplier  │
│   - health_risk_composite│
│   - data_quality_score   │
└──────────┬───────────────┘
           │
           ↓ (Model inference with Feature Store lookup)

STAGE 4: MODEL PREDICTION
┌──────────────────────────┐
│  Random Forest Model     │  
│  - Feature preprocessing │
│  - Risk score prediction │
│  - Model: insurance_model│
└──────────┬───────────────┘
           │
           ↓ (Business rules & categorization)

STAGE 5: PREDICTION OUTPUTS
┌──────────────────────────┐
│ ml_patient_predictions   │  40,000 rows (daily predictions)
│   - Risk scores (0-100)  │
│   - Risk categories      │
│   - Confidence bounds    │
│   - Business flags       │
└──────────────────────────┘
```

**Key Tables in the Pipeline**:

| Table | Purpose | Row Count | Update Frequency |
|-------|---------|-----------|------------------|
| `bronze_patients` | Raw patient data | 40,000 | Real-time streaming |
| `dim_patients` | Dimensional patient data (SCD Type 2) | 40,000 | Daily refresh |
| `ml_insurance_features` | Engineered ML features | 10,000 | Daily refresh |
| `ml_patient_predictions` | Final risk predictions | 40,000 | Daily batch (2 AM UTC) |

---

## Stage 1: Raw Patient Data (Bronze Layer)

### Purpose
The bronze layer ingests raw patient data from source systems via streaming pipelines. This is the landing zone for all patient demographics, health metrics, and insurance information before any transformations.

**Source Table**: `juan_dev.healthcare_data.bronze_patients`  
**Type**: STREAMING_TABLE  
**Row Count**: 40,000 records  
**Data Classification**: PHI (Protected Health Information)  
**Compliance**: HIPAA

### Schema Overview

Key columns that flow through to predictions:

| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `patient_id` | STRING | Unique patient identifier | "P00001" |
| `first_name` | STRING | Patient first name (PII) | "John" |
| `last_name` | STRING | Patient last name (PII) | "Smith" |
| `age` | INT | Patient age | 42 |
| `sex` | STRING | Patient gender | "M" |
| `region` | STRING | Geographic region | "NORTHWEST" |
| `bmi` | DOUBLE | Body Mass Index | 28.5 |
| `smoker` | BOOLEAN | Smoking status | true |
| `children` | INT | Number of dependents | 2 |
| `charges` | DOUBLE | Historical insurance charges | $15,234.89 |
| `insurance_plan` | STRING | Insurance plan type | "Gold" |
| `coverage_start_date` | STRING | Coverage start date | "2023-01-15" |
| `_ingested_at` | TIMESTAMP | Ingestion timestamp | 2025-11-22 08:30:00 |
| `_pipeline_env` | STRING | Pipeline environment | "production" |

### Sample Raw Records

**Patient 1 - Low Risk Profile**
```json
{
  "patient_id": "P00001",
  "first_name": "Sarah",
  "last_name": "Johnson",
  "age": 28,
  "sex": "F",
  "region": "SOUTHWEST",
  "bmi": 22.3,
  "smoker": false,
  "children": 0,
  "charges": 3245.67,
  "insurance_plan": "Bronze",
  "coverage_start_date": "2024-03-01",
  "_ingested_at": "2025-11-22T08:30:15Z",
  "_pipeline_env": "production"
}
```

**Patient 2 - Moderate Risk Profile**
```json
{
  "patient_id": "P00782",
  "first_name": "Michael",
  "last_name": "Chen",
  "age": 45,
  "sex": "M",
  "region": "NORTHEAST",
  "bmi": 29.8,
  "smoker": false,
  "children": 2,
  "charges": 8934.21,
  "insurance_plan": "Silver",
  "coverage_start_date": "2022-06-15",
  "_ingested_at": "2025-11-22T08:30:22Z",
  "_pipeline_env": "production"
}
```

**Patient 3 - High Risk Profile**
```json
{
  "patient_id": "P03456",
  "first_name": "Robert",
  "last_name": "Williams",
  "age": 58,
  "sex": "M",
  "region": "SOUTHEAST",
  "bmi": 35.2,
  "smoker": true,
  "children": 3,
  "charges": 42156.89,
  "insurance_plan": "Gold",
  "coverage_start_date": "2020-01-20",
  "_ingested_at": "2025-11-22T08:30:45Z",
  "_pipeline_env": "production"
}
```

**Patient 4 - Senior High Risk Profile**
```json
{
  "patient_id": "P07821",
  "first_name": "Dorothy",
  "last_name": "Martinez",
  "age": 67,
  "sex": "F",
  "region": "NORTHWEST",
  "bmi": 31.5,
  "smoker": false,
  "children": 0,
  "charges": 28456.32,
  "insurance_plan": "Platinum",
  "coverage_start_date": "2019-08-10",
  "_ingested_at": "2025-11-22T08:31:02Z",
  "_pipeline_env": "production"
}
```

### Data Quality Notes

- **Completeness**: All critical fields (age, bmi, smoker, region) have < 1% null rate
- **Validity**: BMI values validated to be within 15-50 range
- **Consistency**: Age values validated to be 18-85
- **Freshness**: Data ingested in real-time with < 5 minute latency
- **Audit Trail**: All records include ingestion metadata for tracking

---

## Stage 2: Feature Engineering

### Overview

Feature engineering transforms raw patient data into ML-ready features that capture health risk patterns. This stage takes the dimensional patient table (`dim_patients`) and creates engineered features that improve model performance.

**Input Table**: `juan_dev.healthcare_data.dim_patients`  
**Output Table**: `juan_dev.healthcare_data.ml_insurance_features`  
**Transformation**: Bronze/Silver → Feature Store  
**Row Count**: 10,000 distinct customers (from 40,000 total records)

### Feature Engineering Process

The feature engineering pipeline:
1. Filters `dim_patients` for current records only (`is_current_record = TRUE`)
2. Maps categorical fields to numeric representations where needed
3. Calculates 6 engineered numerical features
4. Preserves original fields for reference
5. Creates `customer_id` as primary key

### Feature Definitions & Formulas

**Implementation**: See `00-training/00-insurance-model-feature.ipynb` for complete feature engineering code

#### 1. **Age Risk Score** (`age_risk_score`)
A 1-5 scale representing age-related health risk. Age brackets: <25→1, 25-35→2, 35-50→3, 50-65→4, 65+→5

**Rationale**: Older patients typically require more healthcare services and have higher associated costs.

#### 2. **Smoking Impact** (`smoking_impact`)
Amplifies age-related risk for smokers. Calculation: age × 2.5 for smokers, age × 1.0 for non-smokers

**Rationale**: Smoking accelerates age-related health decline by approximately 2.5x based on healthcare research.

#### 3. **Family Size Factor** (`family_size_factor`)
Adjusts risk based on number of dependents. Formula: 1 + (children × 0.15)

**Rationale**: Each dependent increases overall family healthcare utilization by ~15%.

#### 4. **Regional Multiplier** (`regional_multiplier`)
Geographic cost-of-care adjustment. Values: NORTHEAST (1.2), NORTHWEST (1.1), SOUTHEAST (1.0), SOUTHWEST (0.95)

**Rationale**: Healthcare costs vary significantly by region due to provider density, cost of living, and state regulations.

#### 5. **Health Risk Composite** (`health_risk_composite`)
Composite score combining multiple risk factors. Calculation: (age_risk_score × 20) + (smoker: 50) + (bmi > 30: 30)

**Rationale**: Creates a 0-100+ scale that combines age (up to 100 points), smoking status (50 points), and obesity (30 points).

#### 6. **Data Quality Score** (`data_quality_score`)
Derived from source data completeness and validity. Values: 0.0 to 1.0 (1.0 = perfect data quality)

**Source**: Copied from `patient_data_quality_score` field in `dim_patients` table

### Sample Patient Transformation

Let's follow **Patient P00782 (Michael Chen)** through the feature engineering process:

**INPUT: Raw Patient Data (from dim_patients)**
```
patient_natural_key: "P00782"
patient_gender: "M"
patient_age_category: "MIDDLE_AGE"
patient_region: "NORTHEAST"
patient_smoking_status: "NON_SMOKER"
bmi: 29.8
number_of_dependents: 2
health_risk_score: 55.0  (pre-calculated in dim table)
```

**TRANSFORMATION: Feature Engineering Calculations**

1. **Map age category to numeric age**:
   - `patient_age_category = "MIDDLE_AGE"` → `age = 45`

2. **Calculate age_risk_score**:
   - `age = 45` → falls in range `35-50` → `age_risk_score = 3`

3. **Calculate smoking_impact**:
   - `smoker = false` → `smoking_impact = 45 * 1.0 = 45.0`

4. **Calculate family_size_factor**:
   - `children = 2` → `family_size_factor = 1 + (2 * 0.15) = 1.30`

5. **Calculate regional_multiplier**:
   - `region = "NORTHEAST"` → `regional_multiplier = 1.2`

6. **Calculate health_risk_composite**:
   - `age_risk_score = 3` → `3 * 20 = 60`
   - `smoker = false` → `+ 0`
   - `bmi = 29.8` (not > 30) → `+ 0`
   - `health_risk_composite = 60`

7. **Copy data_quality_score**:
   - `data_quality_score = 1.0`

8. **Map additional fields**:
   - `sex = "M"` (from patient_gender)
   - `region = "NORTHEAST"` (preserved)
   - `customer_id = "P00782"` (from patient_natural_key)

**OUTPUT: Feature Record**
```json
{
  "customer_id": "P00782",
  "patient_natural_key": "P00782",
  "patient_surrogate_key": 782,
  
  // Original mapped fields
  "age": 45,
  "sex": "M",
  "region": "NORTHEAST",
  "bmi": 29.8,
  "children": 2,
  "smoker": false,
  
  // Engineered features (used by model)
  "age_risk_score": 3,
  "smoking_impact": 45.0,
  "family_size_factor": 1.30,
  "regional_multiplier": 1.2,
  "health_risk_composite": 60.0,
  "data_quality_score": 1.0,
  
  // Reference fields
  "patient_age_category": "MIDDLE_AGE",
  "patient_smoking_status": "NON_SMOKER",
  "health_risk_score": 55.0,
  "dimension_last_updated": "2025-11-22T03:37:05Z"
}
```

### Sample Feature Records

**Patient 1 - Sarah Johnson (Low Risk)**
```json
{
  "customer_id": "P00001",
  "age": 28,
  "sex": "F",
  "region": "SOUTHWEST",
  "bmi": 22.3,
  "children": 0,
  "smoker": false,
  "age_risk_score": 2,
  "smoking_impact": 28.0,
  "family_size_factor": 1.0,
  "regional_multiplier": 0.95,
  "health_risk_composite": 40.0,
  "data_quality_score": 1.0
}
```

**Patient 2 - Michael Chen (Moderate Risk)**
```json
{
  "customer_id": "P00782",
  "age": 45,
  "sex": "M",
  "region": "NORTHEAST",
  "bmi": 29.8,
  "children": 2,
  "smoker": false,
  "age_risk_score": 3,
  "smoking_impact": 45.0,
  "family_size_factor": 1.30,
  "regional_multiplier": 1.2,
  "health_risk_composite": 60.0,
  "data_quality_score": 1.0
}
```

**Patient 3 - Robert Williams (High Risk - Smoker + Obese)**
```json
{
  "customer_id": "P03456",
  "age": 58,
  "sex": "M",
  "region": "SOUTHEAST",
  "bmi": 35.2,
  "children": 3,
  "smoker": true,
  "age_risk_score": 4,
  "smoking_impact": 145.0,
  "family_size_factor": 1.45,
  "regional_multiplier": 1.0,
  "health_risk_composite": 160.0,
  "data_quality_score": 1.0
}
```

**Patient 4 - Dorothy Martinez (Senior High Risk)**
```json
{
  "customer_id": "P07821",
  "age": 67,
  "sex": "F",
  "region": "NORTHWEST",
  "bmi": 31.5,
  "children": 0,
  "smoker": false,
  "age_risk_score": 5,
  "smoking_impact": 67.0,
  "family_size_factor": 1.0,
  "regional_multiplier": 1.1,
  "health_risk_composite": 130.0,
  "data_quality_score": 1.0
}
```

**Patient 5 - James Lee (Young Adult Low Risk)**
```json
{
  "customer_id": "P09234",
  "age": 23,
  "sex": "M",
  "region": "NORTHWEST",
  "bmi": 24.1,
  "children": 0,
  "smoker": false,
  "age_risk_score": 1,
  "smoking_impact": 23.0,
  "family_size_factor": 1.0,
  "regional_multiplier": 1.1,
  "health_risk_composite": 20.0,
  "data_quality_score": 1.0
}
```

### Feature Table Schema

The `ml_insurance_features` table contains:
- **All original columns** from `dim_patients` (for reference)
- **6 engineered numerical features** (calculated above)
- **Mapped features** (age, sex, region, bmi, children, smoker)
- **Primary key**: `customer_id`

**Model-Ready Features** (8 features used by the model):
1. `age_risk_score` (int)
2. `smoking_impact` (decimal)
3. `family_size_factor` (decimal)
4. `health_risk_composite` (double)
5. `regional_multiplier` (decimal)
6. `data_quality_score` (double)
7. `sex` (string → LabelEncoded during inference)
8. `region` (string → LabelEncoded during inference)

---

## Stage 3: Model Prediction Process

### Overview

The model prediction stage uses a trained Random Forest Regressor to generate health risk scores (0-100 scale) for each patient. The Feature Store automatically looks up features, applies preprocessing, and generates predictions.

**Model**: `juan_dev.healthcare_data.insurance_model`  
**Model Type**: Random Forest Regressor  
**Algorithm**: sklearn.ensemble.RandomForestRegressor  
**Target Variable**: `health_risk_score` (0-100 scale)

**Implementation**: See `02-batch/insurance-model-batch.ipynb` for complete inference pipeline

### Model Architecture

**Hyperparameters**: 100 decision trees, max depth 10, min samples split 5, min samples leaf 2

**Performance Metrics** (Validation Set):
- R² Score: 0.75
- Mean Absolute Error (MAE): 12.3
- High-Risk Accuracy: 68%

**Training Details**: See `00-training/01-insurance-model-train.ipynb` for model training configuration

### Feature Preprocessing

Before prediction, features undergo preprocessing embedded in the model:

**1. Label Encoding (Categorical Features)**:
- `sex`: M→0, F→1
- `region`: NORTHEAST→0, NORTHWEST→1, SOUTHEAST→2, SOUTHWEST→3

**2. Standard Scaling (All Features)**:
- All 8 features are z-score normalized using parameters fitted on training data
- Formula: `(value - mean) / std_dev`

**Implementation**: See `HealthcareRiskPipeline` class in `00-training/01-insurance-model-train.ipynb`

### Inference Process with Feature Store

**Implementation**: See `HealthcareBatchInference.run_batch_inference()` in `02-batch/insurance-model-batch.ipynb`

**Step-by-Step Prediction Flow**:

1. **Input Preparation**: Load patient data from `dim_patients` (current records only) and create `customer_id` mapping
2. **Feature Store Lookup**: Feature Engineering Client automatically joins with `ml_insurance_features` and fetches all 8 required features
3. **Model Preprocessing**: Embedded pipeline applies label encoding and standard scaling
4. **Prediction Generation**: Random Forest ensemble (100 trees) generates predictions
5. **Output**: Raw risk score (0-100 scale)

### Sample Predictions

**Patient 1 - Sarah Johnson (Low Risk)**

**Input Features**:
```json
{
  "age_risk_score": 2,
  "smoking_impact": 28.0,
  "family_size_factor": 1.0,
  "health_risk_composite": 40.0,
  "regional_multiplier": 0.95,
  "data_quality_score": 1.0,
  "sex": "F",           // → Encoded as 1
  "region": "SOUTHWEST" // → Encoded as 3
}
```

**After Preprocessing** (scaled features):
```json
{
  "age_risk_score_scaled": -0.82,
  "smoking_impact_scaled": -1.15,
  "family_size_factor_scaled": -0.94,
  "health_risk_composite_scaled": -0.98,
  "regional_multiplier_scaled": -0.77,
  "data_quality_score_scaled": 0.0,
  "sex_encoded_scaled": 0.45,
  "region_encoded_scaled": 1.23
}
```

**Model Output**:
```json
{
  "prediction": 28.5,  // Raw model prediction (0-100 scale)
  "model_version": "1",
  "model_name": "juan_dev.healthcare_data.insurance_model"
}
```

---

**Patient 2 - Michael Chen (Moderate Risk)**

**Input Features**:
```json
{
  "age_risk_score": 3,
  "smoking_impact": 45.0,
  "family_size_factor": 1.30,
  "health_risk_composite": 60.0,
  "regional_multiplier": 1.2,
  "data_quality_score": 1.0,
  "sex": "M",          // → Encoded as 0
  "region": "NORTHEAST" // → Encoded as 0
}
```

**Model Output**:
```json
{
  "prediction": 56.8,  // Moderate risk score
  "model_version": "1",
  "model_name": "juan_dev.healthcare_data.insurance_model"
}
```

---

**Patient 3 - Robert Williams (High Risk)**

**Input Features**:
```json
{
  "age_risk_score": 4,
  "smoking_impact": 145.0,
  "family_size_factor": 1.45,
  "health_risk_composite": 160.0,
  "regional_multiplier": 1.0,
  "data_quality_score": 1.0,
  "sex": "M",
  "region": "SOUTHEAST"
}
```

**Model Output**:
```json
{
  "prediction": 78.3,  // High risk score
  "model_version": "1",
  "model_name": "juan_dev.healthcare_data.insurance_model"
}
```

---

**Patient 4 - Dorothy Martinez (Senior High Risk)**

**Input Features**:
```json
{
  "age_risk_score": 5,
  "smoking_impact": 67.0,
  "family_size_factor": 1.0,
  "health_risk_composite": 130.0,
  "regional_multiplier": 1.1,
  "data_quality_score": 1.0,
  "sex": "F",
  "region": "NORTHWEST"
}
```

**Model Output**:
```json
{
  "prediction": 71.2,  // High risk (age-driven)
  "model_version": "1",
  "model_name": "juan_dev.healthcare_data.insurance_model"
}
```

---

**Patient 5 - James Lee (Young Adult Low Risk)**

**Input Features**:
```json
{
  "age_risk_score": 1,
  "smoking_impact": 23.0,
  "family_size_factor": 1.0,
  "health_risk_composite": 20.0,
  "regional_multiplier": 1.1,
  "data_quality_score": 1.0,
  "sex": "M",
  "region": "NORTHWEST"
}
```

**Model Output**:
```json
{
  "prediction": 22.7,  // Very low risk
  "model_version": "1",
  "model_name": "juan_dev.healthcare_data.insurance_model"
}
```

### Feature Store Integration Benefits

- **Automatic Feature Lookup**: No manual joining required during inference
- **Consistency**: Same features used in training and inference
- **Versioning**: Feature definitions tracked with model
- **Performance**: Optimized joins and caching
- **Governance**: Lineage tracking from features → predictions

---

## Stage 4: Prediction Outputs & Business Rules

### Overview

After the model generates raw predictions, business logic is applied to create actionable outputs. This stage adds risk categorization, confidence bounds, and decision flags.

**Output Table**: `juan_dev.healthcare_data.ml_patient_predictions`  
**Row Count**: 40,000 (daily batch predictions)  
**Update Frequency**: Daily at 2:00 AM UTC

### Business Rules Applied

**Implementation**: See `HealthcareBatchInference.run_batch_inference()` in `02-batch/insurance-model-batch.ipynb`

#### 1. **Minimum Risk Threshold**
No patient can have a risk score below 10 (business policy). Formula: `GREATEST(prediction, 10)`

#### 2. **Risk Categorization**
Thresholds: low (<30), medium (30-60), high (60-85), critical (≥85)

#### 3. **Confidence Intervals**
Lower bound: prediction × 0.90 (-10%)  
Upper bound: prediction × 1.10 (+10%)

#### 4. **Business Flags**
- `high_risk_patient`: True if score > 75 OR category = 'critical'
- `requires_review`: True if score > 90

### Sample Output Records

**Patient 1 - Sarah Johnson (Low Risk)**
```json
{
  "customer_id": "P00001",
  "patient_natural_key": "P00001",
  
  // Model predictions
  "prediction": 28.5,
  "adjusted_prediction": 28.5,
  
  // Risk categorization
  "risk_category": "low",
  
  // Confidence bounds
  "prediction_lower_bound": 25.7,
  "prediction_upper_bound": 31.4,
  
  // Business flags
  "high_risk_patient": false,
  "requires_review": false,
  
  // Metadata
  "prediction_timestamp": "2025-11-22T02:15:23Z",
  "model_version": "1",
  "model_name": "juan_dev.healthcare_data.insurance_model"
}
```

**Interpretation**: Young, healthy patient with low healthcare needs. Standard enrollment, base premium tier.

---

**Patient 2 - Michael Chen (Medium Risk)**
```json
{
  "customer_id": "P00782",
  "patient_natural_key": "P00782",
  
  "prediction": 56.8,
  "adjusted_prediction": 56.8,
  
  "risk_category": "medium",
  
  "prediction_lower_bound": 51.1,
  "prediction_upper_bound": 62.5,
  
  "high_risk_patient": false,
  "requires_review": false,
  
  "prediction_timestamp": "2025-11-22T02:15:23Z",
  "model_version": "1",
  "model_name": "juan_dev.healthcare_data.insurance_model"
}
```

**Interpretation**: Middle-aged patient with moderate risk factors. Standard enrollment, moderate premium adjustment. Consider wellness program enrollment.

---

**Patient 3 - Robert Williams (High Risk - Smoker + Obese)**
```json
{
  "customer_id": "P03456",
  "patient_natural_key": "P03456",
  
  "prediction": 78.3,
  "adjusted_prediction": 78.3,
  
  "risk_category": "high",
  
  "prediction_lower_bound": 70.5,
  "prediction_upper_bound": 86.1,
  
  "high_risk_patient": true,   // ← Flag triggered (score > 75)
  "requires_review": false,
  
  "prediction_timestamp": "2025-11-22T02:15:23Z",
  "model_version": "1",
  "model_name": "juan_dev.healthcare_data.insurance_model"
}
```

**Interpretation**: High-risk patient due to smoking and obesity. Automatically enrolled in care management program. Premium adjusted upward. Proactive outreach for smoking cessation and weight management.

---

**Patient 4 - Dorothy Martinez (High Risk - Senior)**
```json
{
  "customer_id": "P07821",
  "patient_natural_key": "P07821",
  
  "prediction": 71.2,
  "adjusted_prediction": 71.2,
  
  "risk_category": "high",
  
  "prediction_lower_bound": 64.1,
  "prediction_upper_bound": 78.3,
  
  "high_risk_patient": false,  // Just below 75 threshold
  "requires_review": false,
  
  "prediction_timestamp": "2025-11-22T02:15:23Z",
  "model_version": "1",
  "model_name": "juan_dev.healthcare_data.insurance_model"
}
```

**Interpretation**: Senior patient with elevated risk but not flagged for intensive intervention. Monthly wellness checks recommended.

---

**Patient 5 - James Lee (Low Risk - Young Adult)**
```json
{
  "customer_id": "P09234",
  "patient_natural_key": "P09234",
  
  "prediction": 22.7,
  "adjusted_prediction": 22.7,
  
  "risk_category": "low",
  
  "prediction_lower_bound": 20.4,
  "prediction_upper_bound": 25.0,
  
  "high_risk_patient": false,
  "requires_review": false,
  
  "prediction_timestamp": "2025-11-22T02:15:23Z",
  "model_version": "1",
  "model_name": "juan_dev.healthcare_data.insurance_model"
}
```

**Interpretation**: Very low risk. Expedited enrollment, lowest premium tier, annual checkups only.

---

**Patient 6 - Critical Risk Example**
```json
{
  "customer_id": "P05689",
  "patient_natural_key": "P05689",
  
  "prediction": 92.4,
  "adjusted_prediction": 92.4,
  
  "risk_category": "critical",
  
  "prediction_lower_bound": 83.2,
  "prediction_upper_bound": 101.6,
  
  "high_risk_patient": true,
  "requires_review": true,     // ← Manual review required (score > 90)
  
  "prediction_timestamp": "2025-11-22T02:15:23Z",
  "model_version": "1",
  "model_name": "juan_dev.healthcare_data.insurance_model"
}
```

**Interpretation**: Critical risk patient requiring immediate attention. Routed to senior underwriter for manual review. Intensive case management recommended. May require specialized care coordination.

---

**Patient 7 - Edge Case: Very Low Raw Prediction**
```json
{
  "customer_id": "P08234",
  "patient_natural_key": "P08234",
  
  "prediction": 8.2,            // Raw prediction below threshold
  "adjusted_prediction": 10.0,  // ← Adjusted to minimum (business rule)
  
  "risk_category": "low",
  
  "prediction_lower_bound": 9.0,
  "prediction_upper_bound": 11.0,
  
  "high_risk_patient": false,
  "requires_review": false,
  
  "prediction_timestamp": "2025-11-22T02:15:23Z",
  "model_version": "1",
  "model_name": "juan_dev.healthcare_data.insurance_model"
}
```

**Interpretation**: Extremely healthy young patient. Adjusted to minimum risk score of 10 per business policy (all patients have some baseline risk).

---

### Output Schema Summary

| Column | Data Type | Description |
|--------|-----------|-------------|
| `customer_id` | STRING | Customer identifier (primary key) |
| `patient_natural_key` | STRING | Original patient ID |
| `prediction` | DOUBLE | Raw model output (0-100) |
| `adjusted_prediction` | DOUBLE | Business-adjusted score (minimum 10) |
| `risk_category` | STRING | low/medium/high/critical |
| `prediction_lower_bound` | DOUBLE | Lower confidence bound (-10%) |
| `prediction_upper_bound` | DOUBLE | Upper confidence bound (+10%) |
| `high_risk_patient` | BOOLEAN | Flag for care management enrollment |
| `requires_review` | BOOLEAN | Flag for manual underwriter review |
| `prediction_timestamp` | TIMESTAMP | When prediction was generated |
| `model_version` | STRING | Model version used |
| `model_name` | STRING | Full model name |

Plus all original fields from `dim_patients` and `ml_insurance_features` for reference and downstream analysis.

---

## Complete End-to-End Example

Let's follow **Patient P00782 (Michael Chen)** through the entire pipeline from raw data to final prediction output.

### Step 1: Raw Data Input (Bronze Layer)

**Source**: Healthcare system ingests patient enrollment data

```json
{
  "patient_id": "P00782",
  "first_name": "Michael",
  "last_name": "Chen",
  "age": 45,
  "sex": "M",
  "region": "NORTHEAST",
  "bmi": 29.8,
  "smoker": false,
  "children": 2,
  "charges": 8934.21,
  "insurance_plan": "Silver",
  "coverage_start_date": "2022-06-15",
  "_ingested_at": "2025-11-22T08:30:22Z",
  "_pipeline_env": "production"
}
```

**Status**: Raw data in `bronze_patients` table

---

### Step 2: Data Cleansing & Transformation

**Process**: Bronze → Silver → Gold (Dimensional Model)

**Silver Layer** (`silver_patients`):
- Validate data quality (no nulls, BMI in valid range)
- Add quality scores
- De-identify sensitive fields

**Gold Layer** (`dim_patients`):
- Create SCD Type 2 record
- Enrich with demographic segments
- Calculate health risk category

```json
{
  "patient_natural_key": "P00782",
  "patient_surrogate_key": 782,
  "patient_gender": "M",
  "patient_age_category": "MIDDLE_AGE",
  "patient_region": "NORTHEAST",
  "patient_smoking_status": "NON_SMOKER",
  "bmi": 29.8,
  "number_of_dependents": 2,
  "health_risk_category": "MODERATE_RISK",
  "health_risk_score": 55.0,
  "is_current_record": true,
  "effective_date": "2022-06-15T00:00:00Z",
  "expiry_date": null,
  "patient_data_quality_score": 1.0
}
```

**Status**: Dimensional data ready for feature engineering

---

### Step 3: Feature Engineering

**Process**: Calculate 6 engineered features

**Age Mapping**:
```
patient_age_category = "MIDDLE_AGE" → age = 45
```

**Feature Calculations**:

1. **age_risk_score**: `age = 45` falls in 35-50 range → `3`

2. **smoking_impact**: `smoker = false` → `45 * 1.0 = 45.0`

3. **family_size_factor**: `children = 2` → `1 + (2 * 0.15) = 1.30`

4. **regional_multiplier**: `region = "NORTHEAST"` → `1.2`

5. **health_risk_composite**: 
   ```
   (age_risk_score * 20) + (smoker ? 50 : 0) + (bmi > 30 ? 30 : 0)
   = (3 * 20) + 0 + 0
   = 60.0
   ```

6. **data_quality_score**: `1.0` (copied from source)

**Output to Feature Store** (`ml_insurance_features`):
```json
{
  "customer_id": "P00782",
  "age": 45,
  "sex": "M",
  "region": "NORTHEAST",
  "bmi": 29.8,
  "children": 2,
  "smoker": false,
  "age_risk_score": 3,
  "smoking_impact": 45.0,
  "family_size_factor": 1.30,
  "regional_multiplier": 1.2,
  "health_risk_composite": 60.0,
  "data_quality_score": 1.0
}
```

**Status**: ML-ready features in Feature Store

---

### Step 4: Model Prediction

**Process**: Feature Store lookup → Preprocessing → Model inference

**4a. Feature Store Lookup**:
- Input: `customer_id = "P00782"`
- Feature Store automatically joins and fetches all 8 features

**4b. Preprocessing**:

Label Encoding:
```
sex = "M" → 0
region = "NORTHEAST" → 0
```

Standard Scaling (example for one feature):
```
age_risk_score = 3
scaled = (3 - mean_age_risk_score) / std_age_risk_score
scaled ≈ -0.25  (example)
```
*All 8 features are scaled using training distribution parameters*

**4c. Model Inference**:
- Random Forest with 100 trees
- Each tree produces a prediction
- Final prediction = average across all trees

**Raw Model Output**:
```json
{
  "prediction": 56.8
}
```

**Status**: Raw risk score generated (0-100 scale)

---

### Step 5: Business Rules Application

**Process**: Apply business logic and create actionable outputs

**5a. Adjusted Prediction**:
```
adjusted_prediction = GREATEST(56.8, 10) = 56.8
```
(No adjustment needed; already above minimum)

**5b. Risk Category**:
```
56.8 is in range [30, 60) → "medium"
```

**5c. Confidence Bounds**:
```
lower_bound = 56.8 * 0.90 = 51.1
upper_bound = 56.8 * 1.10 = 62.5
```

**5d. Business Flags**:
```
high_risk_patient = (56.8 > 75) OR (category = "critical") = false
requires_review = (56.8 > 90) = false
```

**5e. Add Metadata**:
```
prediction_timestamp = "2025-11-22T02:15:23Z"
model_version = "1"
model_name = "juan_dev.healthcare_data.insurance_model"
```

**Final Output** (`ml_patient_predictions`):
```json
{
  "customer_id": "P00782",
  "patient_natural_key": "P00782",
  "patient_surrogate_key": 782,
  
  // Original patient data (for reference)
  "patient_age_category": "MIDDLE_AGE",
  "patient_gender": "M",
  "patient_region": "NORTHEAST",
  "bmi": 29.8,
  "patient_smoking_status": "NON_SMOKER",
  "number_of_dependents": 2,
  
  // Engineered features (for analysis)
  "age": 45,
  "sex": "M",
  "region": "NORTHEAST",
  "age_risk_score": 3,
  "smoking_impact": 45.0,
  "family_size_factor": 1.30,
  "regional_multiplier": 1.2,
  "health_risk_composite": 60.0,
  "data_quality_score": 1.0,
  
  // Predictions
  "prediction": 56.8,
  "adjusted_prediction": 56.8,
  
  // Business outputs
  "risk_category": "medium",
  "prediction_lower_bound": 51.1,
  "prediction_upper_bound": 62.5,
  "high_risk_patient": false,
  "requires_review": false,
  
  // Metadata
  "prediction_timestamp": "2025-11-22T02:15:23Z",
  "model_version": "1",
  "model_name": "juan_dev.healthcare_data.insurance_model"
}
```

**Status**: Complete prediction record ready for downstream systems

---

### Step 6: Downstream Use

**Care Management System**:
- Query: `SELECT * FROM ml_patient_predictions WHERE high_risk_patient = true`
- Result: Michael Chen NOT included (moderate risk only)
- Action: No immediate intervention, but quarterly wellness check recommended

**Premium Calculation Engine**:
- Input: `risk_category = "medium"`, `adjusted_prediction = 56.8`
- Calculation: `base_premium × regional_multiplier × risk_multiplier`
- Output: Monthly premium = $450 (moderate tier)

**BI Dashboard**:
- Aggregate: Risk distribution by region
- Michael Chen contributes to: "NORTHEAST, medium risk" count
- Used for: Resource allocation, trend analysis

---

## Interpreting Prediction Outputs

### Risk Score Scale (0-100)

The adjusted prediction is a continuous score representing expected healthcare utilization and cost:

| Score Range | Risk Level | Expected Healthcare Utilization | Annual Estimated Cost |
|-------------|------------|--------------------------------|----------------------|
| 0-29 | Low | Minimal - Preventive care only | < $5,000 |
| 30-59 | Medium | Moderate - Routine care, some interventions | $5,000 - $15,000 |
| 60-84 | High | Elevated - Active management needed | $15,000 - $35,000 |
| 85-100 | Critical | Very High - Intensive intervention | > $35,000 |

### Risk Category Meanings

**Low Risk** (`score < 30`):
- **Clinical**: Healthy baseline, no significant risk factors
- **Care Plan**: Annual checkups, preventive screenings
- **Business Action**: Standard enrollment, base premium tier
- **Typical Profile**: Young adults, healthy lifestyle, no chronic conditions

**Medium Risk** (`30 ≤ score < 60`):
- **Clinical**: Some risk factors present, moderate intervention needs
- **Care Plan**: Quarterly wellness checks, lifestyle coaching
- **Business Action**: Moderate premium adjustment, wellness program eligibility
- **Typical Profile**: Middle-aged, some health concerns (overweight, family history)

**High Risk** (`60 ≤ score < 85`):
- **Clinical**: Multiple risk factors, chronic condition management
- **Care Plan**: Monthly care management calls, care coordinator assigned
- **Business Action**: Higher premium tier, care management enrollment
- **Typical Profile**: Seniors, smokers with health issues, obese with comorbidities

**Critical Risk** (`score ≥ 85`):
- **Clinical**: Severe risk profile, potential for high-cost events
- **Care Plan**: Weekly interventions, intensive case management
- **Business Action**: Manual underwriting review, specialized care coordination
- **Typical Profile**: Multiple chronic conditions, history of hospitalizations, severe lifestyle risks

### Confidence Bounds Interpretation

**Lower Bound** (`prediction × 0.90`):
- Conservative estimate for budgeting
- Minimum expected risk/cost

**Upper Bound** (`prediction × 1.10`):
- Optimistic estimate for capacity planning
- Maximum expected risk/cost within normal variation

**Use Case**:
- **Financial Planning**: Use lower bound for minimum reserve calculations
- **Capacity Planning**: Use upper bound for maximum care management resources needed
- **Pricing**: Use midpoint (adjusted_prediction) for premium calculations

**Example** (Michael Chen):
```
adjusted_prediction = 56.8
lower_bound = 51.1  (10% lower)
upper_bound = 62.5  (10% higher)

Interpretation: Expected cost $8,000-$10,000 annually with 90% confidence
```

### Business Flags

**`high_risk_patient` Flag**:
- **Trigger**: `adjusted_prediction > 75` OR `risk_category = 'critical'`
- **Action**: Automatic enrollment in care management program
- **Frequency**: Daily batch identifies ~23% of population (9,500 patients)
- **Outcome**: 15-20% reduction in preventable hospitalizations

**`requires_review` Flag**:
- **Trigger**: `adjusted_prediction > 90`
- **Action**: Route to senior underwriter for manual review
- **Frequency**: ~5% of daily predictions
- **Purpose**: Ensure appropriate coverage and intervention for highest-risk patients

### Decision Making Guidelines

**For Care Managers**:
1. Prioritize patients with `high_risk_patient = true`
2. Within high-risk, prioritize by `adjusted_prediction` (highest first)
3. Review `requires_review` patients daily
4. Track risk category changes month-over-month

**For Underwriters**:
1. `requires_review = true` → Manual assessment required
2. `risk_category = 'critical'` → Consider specialized policy terms
3. Review confidence bounds for pricing flexibility
4. Historical trend analysis for existing customers

**For Business Analysts**:
1. Monitor risk distribution shifts (drift detection)
2. Compare predicted vs. actual costs quarterly
3. Analyze regional variations in risk profiles
4. Track care management ROI by risk category

---

## Common Use Cases

### Use Case 1: New Patient Enrollment

**Scenario**: A new patient completes online enrollment

**Data Flow**:
1. Patient submits application with demographics and health info
2. Data ingested into `bronze_patients` table
3. Daily pipeline processes new records:
   - Transform to `dim_patients`
   - Engineer features in `ml_insurance_features`
   - Generate prediction in `ml_patient_predictions`
4. Premium calculation engine queries predictions table
5. Patient receives quote within 24 hours

**Example**:
```
Patient: Sarah Johnson, 28F, SOUTHWEST, BMI 22.3, non-smoker, no children
↓
Prediction: 28.5 (low risk)
↓
Premium: $250/month (base tier)
↓
Action: Expedited enrollment, no underwriting review needed
```

**Business Impact**:
- Consistent pricing across all applicants
- Automated decision for 95% of enrollments
- 24-hour turnaround vs. 3-5 days manual process

---

### Use Case 2: Daily Care Management Assignment

**Scenario**: Scheduled daily batch at 2 AM UTC identifies high-risk patients for proactive intervention

**Data Flow**:
1. Batch inference job runs on all active patients (40,000 records)
2. Predictions written to `ml_patient_predictions` table
3. Care management system queries for high-risk patients:
   ```sql
   SELECT customer_id, adjusted_prediction, risk_category
   FROM ml_patient_predictions
   WHERE high_risk_patient = true
     AND prediction_timestamp >= CURRENT_DATE
   ORDER BY adjusted_prediction DESC;
   ```
4. Results exported to care management platform (Salesforce Health Cloud)
5. Care managers receive daily assignment lists by region

**Example Output**:
```
High-Risk Patients Today (2025-11-22):
- P03456 (Robert Williams): 78.3 - SOUTHEAST - Smoking cessation outreach
- P07821 (Dorothy Martinez): 71.2 - NORTHWEST - Senior wellness check
- P05689 (Critical case): 92.4 - NORTHEAST - Urgent case manager assignment
... (9,500 total)
```

**Business Impact**:
- Proactive intervention for 23% of population
- 15-20% reduction in emergency hospitalizations
- $1,200+ annual savings per managed patient
- Improved patient outcomes and satisfaction

---

### Use Case 3: Population Health Analytics

**Scenario**: Monthly executive review of risk distribution and trends

**Data Flow**:
1. BI dashboard queries prediction history:
   ```sql
   SELECT 
     patient_region,
     risk_category,
     COUNT(*) as patient_count,
     AVG(adjusted_prediction) as avg_risk_score
   FROM ml_patient_predictions
   WHERE prediction_timestamp >= CURRENT_DATE - INTERVAL 30 DAYS
   GROUP BY patient_region, risk_category
   ORDER BY patient_region, avg_risk_score DESC;
   ```
2. Compare current month vs. previous months
3. Identify regional trends and outliers
4. Resource allocation decisions based on risk concentration

**Example Insights**:
```
Regional Risk Distribution (November 2025):

NORTHEAST:
- High/Critical: 5,200 patients (52%) - ALERT: 10% increase vs. October
- Medium: 2,800 patients (28%)
- Low: 2,000 patients (20%)
- Action: Allocate 2 additional care managers to NORTHEAST region

SOUTHWEST:
- High/Critical: 3,800 patients (38%)
- Medium: 3,200 patients (32%)
- Low: 3,000 patients (30%)
- Action: Maintain current staffing
```

**Business Impact**:
- Data-driven resource allocation
- Early detection of population health trends
- Optimized care management staffing by region
- Budget planning aligned with risk distribution

---

## Monitoring & Quality

### How to Check Prediction Quality

**Daily Monitoring Queries**:

1. **Prediction Volume**: Query daily prediction counts by date and model version. Expected: 40,000 predictions daily with consistent model version.

2. **Risk Distribution**: Group by risk category with percentage calculations. Expected distribution: Low (33%), Medium (13%), High (45%), Critical (9%).

3. **Score Statistics**: Calculate average, min, max, and median adjusted predictions. Expected: avg ≈ 53, min = 10, max < 100, median ≈ 50.

**Query Examples**: See `docs/MONITORING.md` for complete monitoring SQL queries and alert thresholds.

### Monitoring Tables

For comprehensive monitoring, query these tables:

**Lakehouse Monitoring (Auto-generated)**:
- `ml_patient_predictions_profile_metrics` - Prediction distributions and statistics
- `ml_patient_predictions_drift_metrics` - Drift detection (PSI, KL divergence)

**Custom Healthcare Metrics**:
- `custom_fairness_metrics` - Fairness metrics by demographics
- `custom_business_metrics` - Business KPIs (risk distribution, throughput)
- `drift_analysis_summary` - Drift analysis summary

**Access**: See `03-monitoring/insurance-model-monitoring.ipynb` for examples using `MonitorAnalyzer` class, or query tables directly in SQL.

### Key Metrics to Track

| Metric | Threshold | Action if Violated |
|--------|-----------|-------------------|
| **Daily Prediction Count** | 35,000 - 45,000 | Check pipeline health |
| **PSI (Population Stability Index)** | < 0.25 | Consider model retraining |
| **Fairness Disparity** | < 15% across demographics | Bias investigation |
| **Null Rate** | < 5% in predictions | Data quality check |
| **High-Risk Rate Shift** | ±10% vs. baseline | Clinical validation |
| **Average Score Drift** | ±5 points vs. baseline | Investigate data changes |

**Alert Configuration**:
- **Critical**: PSI > 0.25 + accuracy degradation → Immediate retraining evaluation
- **High**: Prediction count < 35,000 → Pipeline investigation
- **Medium**: High-risk rate shifts > 30% → Clinical review
- **Low**: Fairness disparity 10-15% → Monitor for trends

**Monitoring Schedule**:
- **Daily**: Automated refresh at 6 AM UTC (after 2 AM batch inference)
- **Weekly**: Data science team reviews drift trends
- **Monthly**: Executive review and governance validation

For complete monitoring documentation, see: [MODEL-MONITORING.md](MODEL-MONITORING.md)

---

## Technical Notes & Tips

### Query Examples to Explore Data

**Common Query Patterns**:

1. **Find Patient's Complete Journey**: Join `dim_patients`, `ml_insurance_features`, and `ml_patient_predictions` on `patient_natural_key`/`customer_id` to see full data flow.

2. **Analyze Feature Impact**: Group predictions by `risk_category` and aggregate feature values to understand correlations.

3. **Identify Edge Cases**: Find patients where `adjusted_prediction` differs from raw `prediction` (business rule adjustments applied).

4. **Risk Category Transitions**: Compare current vs. historical predictions to identify patients whose risk has changed significantly.

**Query Templates**: See `docs/TABLES.md` for detailed table schemas and `docs/MONITORING.md` for monitoring query examples.

### Troubleshooting Common Issues

**Issue 1: Predictions Missing for Some Patients**

**Symptom**: Expected 40,000 predictions but only got 35,000

**Possible Causes**:
- Feature Store missing records for some `customer_id` values
- `is_current_record = false` for some patients (SCD Type 2)
- Data quality issues causing filter exclusions

**Debugging Steps**:
1. Find patients in `dim_patients` (current records) not in predictions
2. Check if missing patients exist in `ml_insurance_features`
3. Run feature engineering job: `00-training/00-insurance-model-feature.ipynb`

**Solution**: Ensure all current patients have corresponding feature records before running batch inference.

---

**Issue 2: All Predictions Are Same Value**

**Symptom**: Every prediction is 53.2 (or some constant)

**Possible Causes**:
- Model loaded incorrectly (using default/fallback)
- Feature Store returning nulls (model uses defaults)
- Preprocessing scaler not applied correctly

**Debugging Steps**:
1. Check feature values in predictions table for variety
2. Verify Feature Store integration in batch inference notebook
3. Check model URI and loading: `models:/juan_dev.healthcare_data.insurance_model@champion`
4. Review preprocessing pipeline in model

**Solution**: If all predictions are identical, feature lookup has failed. Review `02-batch/insurance-model-batch.ipynb` for Feature Store integration.

---

**Issue 3: Risk Distribution Looks Wrong**

**Symptom**: 80% of patients in "high" category (expected: 45%)

**Possible Causes**:
- Model drift (population changed significantly)
- Feature engineering calculation error
- Business rules misconfigured

**Debugging Steps**:
1. Compare current risk distribution to historical (7-day trend)
2. Check average scores by date for sudden shifts
3. Review monitoring metrics for drift alerts

**Solution**: If sudden distribution shift detected, investigate:
- Data quality issues in `dim_patients`
- Population changes (enrollment periods)
- Drift metrics in `drift_analysis_summary` table

**Monitoring**: See `docs/MONITORING.md` for comprehensive drift detection and alerting.

---

### Understanding Null Values and Edge Cases

**Null Handling in Features**:
- Feature engineering filters out records with critical nulls (age, bmi, smoking status)
- If a patient reaches feature table with nulls → data quality score reduced
- Model may produce less confident predictions for low-quality data

**Edge Cases**:

1. **New Patients (< 24 hours in system)**:
   - May not have prediction yet (wait for next daily batch)
   - Check `_ingested_at` timestamp in bronze_patients

2. **SCD Type 2 Historical Records**:
   - Only `is_current_record = true` records get predictions
   - Historical versions preserved for audit but not scored

3. **Extreme Feature Values**:
   - Very high BMI (> 50): Flagged in data quality, may require manual review
   - Very low scores (< 10): Adjusted to minimum of 10 per business rules

4. **Multiple Records Per Patient**:
   - Feature table has ONE record per `customer_id` (deduplicated)
   - Prediction table may have multiple records (daily snapshots)
   - Always use latest `prediction_timestamp` for current risk

---

## Related Documentation

- **[MODEL.md](MODEL.md)** - Complete model architecture, training process, governance, and production deployment
- **[TABLES.md](TABLES.md)** - Detailed schemas, statistics, and data dictionary for all tables
- **[MODEL-MONITORING.md](MODEL-MONITORING.md)** - Comprehensive monitoring infrastructure, alerts, and operational runbooks
- **[README.md](../README.md)** - Project overview and getting started guide

---

## Document Information

**Document Version**: 1.0.0  
**Created**: November 23, 2025  
**Last Updated**: November 23, 2025  
**Maintained By**: Healthcare MLOps Team  
**Purpose**: Educate technical users on the prediction pipeline and data flow

**Target Audience**: Data engineers, analysts, ML engineers, and technical stakeholders

**Changelog**:

### Version 1.0.0 (November 23, 2025)
- Initial creation of predictions user guide
- Documented complete data flow from raw → features → predictions → outputs
- Added sample data at each stage with realistic values
- Included end-to-end patient example (Michael Chen)
- Provided interpretation guidelines for risk scores and categories
- Added common use cases and business workflows
- Included monitoring guidance and query examples
- Added troubleshooting section for common issues

---

**Questions or Feedback?**

For questions about this guide or the prediction system, contact:
- **MLOps Team**: ml-ops@company.com
- **Data Science Team**: data-science@company.com
- **Slack**: #healthcare-ml-ops

**Contributing**:
If you find errors or have suggestions for improving this guide, please submit a pull request or contact the MLOps team.

