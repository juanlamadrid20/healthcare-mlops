# Healthcare Data Tables Documentation

**Catalog:** `juan_dev`  
**Schema:** `healthcare_data`  
**Last Updated:** 2025-11-22  
**Environment:** field-eng-west

## Table Overview

The `juan_dev.healthcare_data` schema contains 14 tables organized across bronze, silver, gold, and ML layers following a medallion architecture pattern.

### Table Summary

| Table Name | Type | Row Count | Purpose | Quality Tier |
|------------|------|-----------|---------|--------------|
| `bronze_patients` | STREAMING_TABLE | 40,000 | Raw patient demographics | Bronze |
| `bronze_claims` | STREAMING_TABLE | 103,368 | Raw insurance claims | Bronze |
| `bronze_medical_events` | STREAMING_TABLE | 338,748 | Raw medical events | Bronze |
| `silver_patients` | MATERIALIZED_VIEW | 40,000 | Cleaned patient data | Silver |
| `silver_claims` | MATERIALIZED_VIEW | 103,368 | Validated claims | Silver |
| `silver_medical_events` | MATERIALIZED_VIEW | 338,748 | Standardized medical events | Silver |
| `dim_patients` | MATERIALIZED_VIEW | 40,000 | Patient dimension (SCD Type 2) | Gold |
| `fact_claims` | MATERIALIZED_VIEW | 413,472 | Claims fact table | Gold |
| `fact_medical_events` | MATERIALIZED_VIEW | 1,219,104 | Medical events fact table | Gold |
| `fact_claims_monthly_summary` | MATERIALIZED_VIEW | 100 | Monthly aggregated claims | Gold |
| `ml_insurance_features` | MANAGED | 10,000 | ML feature store | ML |
| `ml_patient_predictions` | MANAGED | 40,000 | Model predictions | ML |
| `ml_monitoring_summary` | VIEW | 1 | Monitoring summary view | ML |
| `_quality_monitoring_summary` | MANAGED | 696 | Quality monitoring data | System |

---

## Bronze Layer Tables

### 1. bronze_patients

**Type:** STREAMING_TABLE  
**Row Count:** 40,000  
**Purpose:** Raw patient demographics data with HIPAA compliance and audit logging  
**Location:** `s3://databricks-e2demofieldengwest/.../bronze_patients`  
**Data Classification:** PHI  
**Compliance:** HIPAA  
**Pipeline ID:** `a209ea4a-f7a7-4dcf-a822-7c211f7a18f8`

#### Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `patient_id` | string | Patient identifier |
| `first_name` | string | Patient first name (PII) |
| `last_name` | string | Patient last name (PII) |
| `age` | int | Patient age |
| `sex` | string | Patient gender |
| `region` | string | Geographic region |
| `bmi` | double | Body Mass Index |
| `smoker` | boolean | Smoking status |
| `children` | int | Number of dependents |
| `charges` | double | Insurance charges |
| `insurance_plan` | string | Insurance plan type |
| `coverage_start_date` | string | Coverage start date |
| `timestamp` | string | Source timestamp |
| `_file_name` | string | Source file name |
| `_file_path` | string | Source file path |
| `_file_size` | string | Source file size |
| `_file_modification_time` | string | File modification time |
| `_ingested_at` | timestamp | Ingestion timestamp |
| `_pipeline_env` | string | Pipeline environment |
| `_metadata` | struct | File metadata (file_path, file_name, file_size, etc.) |

#### Features
- **Streaming Table:** Supports real-time data ingestion
- **Row Tracking:** Enabled for change data capture
- **Change Data Feed:** Enabled for downstream processing
- **PII Fields:** `first_name`, `last_name` tagged for privacy controls

---

### 2. bronze_claims

**Type:** STREAMING_TABLE  
**Row Count:** 103,368  
**Purpose:** Raw insurance claims data with financial audit trails and referential integrity  
**Location:** `s3://databricks-e2demofieldengwest/.../bronze_claims`  
**Data Classification:** Financial  
**Compliance:** SOX, HIPAA  
**Pipeline ID:** `a209ea4a-f7a7-4dcf-a822-7c211f7a18f8`

#### Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `claim_id` | string | Unique claim identifier |
| `patient_id` | string | Patient identifier (FK) |
| `claim_amount` | double | Claim amount |
| `claim_date` | string | Claim date |
| `diagnosis_code` | string | Medical diagnosis code |
| `procedure_code` | string | Medical procedure code |
| `claim_status` | string | Claim status |
| `timestamp` | string | Source timestamp |
| `_file_name` | string | Source file name |
| `_file_path` | string | Source file path |
| `_file_size` | string | Source file size |
| `_file_modification_time` | string | File modification time |
| `_ingested_at` | timestamp | Ingestion timestamp |
| `_pipeline_env` | string | Pipeline environment |
| `_metadata` | struct | File metadata |

#### Features
- **Streaming Table:** Supports real-time data ingestion
- **Financial Compliance:** SOX compliance tags
- **Row Tracking:** Enabled for audit trails

---

### 3. bronze_medical_events

**Type:** STREAMING_TABLE  
**Row Count:** 338,748  
**Purpose:** Raw medical events data with clinical audit trails and provider validation  
**Location:** `s3://databricks-e2demofieldengwest/.../bronze_medical_events`  
**Data Classification:** Clinical  
**Compliance:** HIPAA  
**Pipeline ID:** `a209ea4a-f7a7-4dcf-a822-7c211f7a18f8`

#### Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `event_id` | string | Unique event identifier |
| `patient_id` | string | Patient identifier (FK) |
| `event_date` | string | Event date |
| `event_type` | string | Type of medical event |
| `medical_provider` | string | Healthcare provider |
| `timestamp` | string | Source timestamp |
| `_file_name` | string | Source file name |
| `_file_path` | string | Source file path |
| `_file_size` | string | Source file size |
| `_file_modification_time` | string | File modification time |
| `_ingested_at` | timestamp | Ingestion timestamp |
| `_pipeline_env` | string | Pipeline environment |
| `_metadata` | struct | File metadata |

#### Features
- **Streaming Table:** Supports real-time data ingestion
- **Clinical Compliance:** HIPAA compliance tags
- **Row Tracking:** Enabled for clinical audit trails

---

## Silver Layer Tables

### 4. silver_patients

**Type:** MATERIALIZED_VIEW  
**Row Count:** 40,000  
**Purpose:** Cleaned and HIPAA-compliant patient data with de-identification and data quality validation  
**Location:** `s3://databricks-e2demofieldengwest/.../silver_patients`  
**Data Classification:** PHI_DEIDENTIFIED  
**Compliance:** HIPAA  
**Last Refreshed:** 2025-11-22T03:37:00Z  
**Total Size:** 453,986 bytes

#### Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `patient_id` | string | Patient identifier |
| `first_name` | string | Patient first name |
| `last_name` | string | Patient last name |
| `age` | int | Patient age |
| `sex` | string | Patient gender |
| `region` | string | Geographic region |
| `bmi` | double | Body Mass Index |
| `smoker` | boolean | Smoking status |
| `children` | int | Number of dependents |
| `charges` | double | Insurance charges |
| `insurance_plan` | string | Insurance plan type |
| `coverage_start_date` | string | Coverage start date |
| `health_risk_category` | string | Health risk category |
| `patient_age_category` | string | Age category |
| `demographic_segment` | string | Demographic segment |
| `patient_data_quality_score` | double | Data quality score |
| `patient_record_quality` | string | Record quality indicator |
| `hipaa_deidentification_applied` | boolean | HIPAA de-identification flag |
| `age_privacy_protection` | boolean | Age privacy protection flag |
| `geographic_privacy_protection` | boolean | Geographic privacy flag |
| `data_retention_compliance` | boolean | Data retention compliance flag |
| `_ingested_at` | timestamp | Ingestion timestamp |
| `_pipeline_env` | string | Pipeline environment |
| `_file_name` | string | Source file name |
| `processed_at` | timestamp | Processing timestamp |

#### Statistical Properties

| Metric | Value |
|--------|-------|
| **Total Rows** | 40,000 |
| **Distinct Patients** | 10,000 |
| **Distinct Regions** | 4 |
| **Age Statistics** | |
| - Average | 44.82 |
| - Minimum | 18 |
| - Maximum | 85 |
| - Nulls | 0 |
| **BMI Statistics** | |
| - Average | 28.07 |
| - Minimum | 16.0 |
| - Maximum | 50.0 |
| - Nulls | 0 |
| **Charges Statistics** | |
| - Average | $6,359.26 |
| - Minimum | $329.55 |
| - Maximum | $55,813.81 |
| - Nulls | 0 |
| **Smoking Status** | |
| - Smokers | 6,592 (16.48%) |
| - Non-smokers | 33,408 (83.52%) |
| **Average Children** | 1.21 |
| **Sex Distribution** | |
| - Male | 20,040 (50.10%) |
| - Female | 19,960 (49.90%) |
| **Region Distribution** | |
| - NORTHEAST | 10,292 (25.73%) |
| - SOUTHWEST | 10,016 (25.04%) |
| - NORTHWEST | 9,916 (24.79%) |
| - SOUTHEAST | 9,776 (24.44%) |
| **Health Risk Category** | |
| - HIGH_RISK | 13,448 (33.62%) |
| - MODERATE_RISK | 13,312 (33.28%) |
| - LOW_RISK | 13,240 (33.10%) |

#### Features
- **HIPAA Compliance:** De-identification applied
- **Data Quality:** Quality scores and validation flags
- **Privacy Controls:** Age and geographic privacy protection
- **Change Data Feed:** Enabled for downstream processing

---

### 5. silver_claims

**Type:** MATERIALIZED_VIEW  
**Row Count:** 103,368  
**Purpose:** Validated insurance claims with referential integrity and financial compliance  
**Location:** `s3://databricks-e2demofieldengwest/.../silver_claims`  
**Data Classification:** Financial  
**Compliance:** SOX, HIPAA  
**Last Refreshed:** 2025-11-22T03:36:58Z  
**Total Size:** 1,272,279 bytes

#### Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `claim_id` | string | Unique claim identifier |
| `patient_id` | string | Patient identifier (FK) |
| `claim_amount_validated` | double | Validated claim amount |
| `claim_date` | string | Claim date |
| `diagnosis_code` | string | Medical diagnosis code |
| `procedure_code` | string | Medical procedure code |
| `claim_status_standardized` | string | Standardized claim status |
| `claim_amount_category` | string | Claim amount category |
| `claim_approved` | boolean | Approval flag |
| `claim_denied` | boolean | Denial flag |
| `claim_paid` | boolean | Payment flag |
| `total_processing_days` | int | Processing duration in days |
| `claim_data_quality_score` | double | Data quality score |
| `_ingested_at` | timestamp | Ingestion timestamp |
| `_pipeline_env` | string | Pipeline environment |
| `_file_name` | string | Source file name |
| `processed_at` | timestamp | Processing timestamp |

#### Statistical Properties

| Metric | Value |
|--------|-------|
| **Total Rows** | 103,368 |
| **Distinct Patients** | 10,000 |
| **Distinct Claims** | 25,842 |
| **Claim Amount Statistics** | |
| - Average | $1,218.94 |
| - Minimum | $50.00 |
| - Maximum | $50,000.00 |
| **Claim Status** | |
| - Approved | 82,552 (79.85%) |
| - Denied | 10,740 (10.40%) |
| **Average Processing Days** | 376.21 |
| **Average Data Quality Score** | 1.0 |

#### Features
- **Financial Compliance:** SOX compliance
- **Referential Integrity:** Validated patient references
- **Status Standardization:** Normalized claim statuses
- **Processing Metrics:** Processing time tracking

---

### 6. silver_medical_events

**Type:** MATERIALIZED_VIEW  
**Row Count:** 338,748  
**Purpose:** Standardized medical events with clinical validation and care coordination metrics  
**Location:** `s3://databricks-e2demofieldengwest/.../silver_medical_events`  
**Data Classification:** Clinical  
**Compliance:** HIPAA  
**Last Refreshed:** 2025-11-22T03:36:59Z  
**Total Size:** 5,477,778 bytes

#### Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `event_id` | string | Unique event identifier |
| `patient_id` | string | Patient identifier (FK) |
| `event_date` | string | Event date |
| `event_type_standardized` | string | Standardized event type |
| `medical_provider` | string | Healthcare provider |
| `emergency_visit` | boolean | Emergency visit indicator |
| `preventive_care_indicator` | boolean | Preventive care flag |
| `acute_care_indicator` | boolean | Acute care flag |
| `hospital_admission` | boolean | Hospital admission flag |
| `provider_type` | string | Provider type classification |
| `facility_type` | string | Facility type classification |
| `clinical_outcome_score` | double | Clinical outcome score |
| `care_efficiency_score` | double | Care efficiency score |
| `care_appropriateness_score` | double | Care appropriateness score |
| `visit_duration_minutes` | int | Visit duration |
| `follow_up_required` | boolean | Follow-up required flag |
| `chronic_management_indicator` | boolean | Chronic management flag |
| `is_new_provider` | boolean | New provider flag |
| `provider_patient_familiarity` | double | Provider-patient familiarity score |
| `days_since_event` | int | Days since event |
| `days_to_next_event` | int | Days to next event |
| `days_since_previous_event` | int | Days since previous event |
| `event_data_quality_score` | double | Data quality score |
| `_ingested_at` | timestamp | Ingestion timestamp |
| `_pipeline_env` | string | Pipeline environment |
| `_file_name` | string | Source file name |
| `processed_at` | timestamp | Processing timestamp |

#### Statistical Properties

| Metric | Value |
|--------|-------|
| **Total Rows** | 338,748 |
| **Distinct Patients** | 10,555 |
| **Distinct Events** | 84,687 |
| **Distinct Event Types** | 824 |
| **Event Categories** | |
| - Emergency Visits | 20,084 (5.93%) |
| - Hospital Admissions | 30,260 (8.93%) |
| **Clinical Scores** | |
| - Average Clinical Outcome Score | 29.99 |
| - Average Care Efficiency Score | 84.99 |
| **Average Data Quality Score** | 1.0 |

#### Features
- **Clinical Validation:** Standardized event types and classifications
- **Care Coordination:** Provider-patient familiarity metrics
- **Clinical Analytics:** Outcome and efficiency scores
- **Temporal Analysis:** Event sequencing and timing metrics

---

## Gold Layer Tables

### 7. dim_patients

**Type:** MATERIALIZED_VIEW  
**Row Count:** 40,000  
**Purpose:** SCD Type 2 patient dimension with comprehensive analytics attributes for business intelligence  
**Location:** `s3://databricks-e2demofieldengwest/.../dim_patients`  
**Data Classification:** ANALYTICS  
**Dimensional Model:** patient_360  
**Last Refreshed:** 2025-11-22T03:37:05Z  
**Total Size:** 1,997,876 bytes

#### Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `patient_surrogate_key` | bigint | Surrogate key (SK) |
| `patient_natural_key` | string | Natural key (NK) |
| `patient_id_hash` | string | Hashed patient ID |
| `patient_name_hash` | string | Hashed patient name |
| `effective_date` | timestamp | SCD effective date |
| `expiry_date` | timestamp | SCD expiry date |
| `is_current_record` | boolean | Current record flag |
| `record_version` | int | Record version number |
| `patient_age_category` | string | Age category |
| `patient_gender` | string | Patient gender |
| `patient_region` | string | Geographic region |
| `demographic_segment` | string | Demographic segment |
| `health_risk_category` | string | Health risk category |
| `health_risk_score` | double | Health risk score (0-100) |
| `patient_smoking_status` | string | Smoking status |
| `bmi` | double | Body Mass Index |
| `obesity_indicator` | boolean | Obesity indicator |
| `underweight_indicator` | boolean | Underweight indicator |
| `lifestyle_risk_factors` | string | Lifestyle risk factors |
| `number_of_dependents` | int | Number of dependents |
| `has_dependents` | boolean | Has dependents flag |
| `family_size_category` | string | Family size category |
| `primary_insurance_plan` | string | Primary insurance plan |
| `coverage_effective_date` | string | Coverage effective date |
| `base_premium` | double | Base premium amount |
| `estimated_annual_premium` | double | Estimated annual premium |
| `patient_premium_category` | string | Premium category |
| `patient_data_quality_score` | double | Data quality score |
| `data_quality_tier` | string | Data quality tier |
| `patient_record_quality` | string | Record quality indicator |
| `hipaa_deidentification_applied` | boolean | HIPAA de-identification flag |
| `age_privacy_protection` | boolean | Age privacy protection flag |
| `geographic_privacy_protection` | boolean | Geographic privacy flag |
| `data_retention_compliance` | boolean | Data retention compliance flag |
| `_pipeline_env` | string | Pipeline environment |
| `dimension_last_updated` | timestamp | Last update timestamp |

#### Statistical Properties

| Metric | Value |
|--------|-------|
| **Total Rows** | 40,000 |
| **Distinct Patients** | 10,000 |
| **Distinct Regions** | 4 |
| **Current Records** | 40,000 (100%) |
| **Health Risk Score** | |
| - Average | 53.14 |
| - Minimum | 25.0 |
| - Maximum | 85.0 |
| **BMI Statistics** | |
| - Average | 28.07 |
| **Premium Statistics** | |
| - Average Base Premium | $6,359.26 |
| - Average Estimated Annual Premium | $8,461.09 |
| **Indicators** | |
| - Obesity Cases | 15,004 (37.51%) |
| - Has Dependents | 28,096 (70.24%) |
| **Average Data Quality Score** | 1.0 |

#### Features
- **SCD Type 2:** Historical tracking with effective/expiry dates
- **Surrogate Keys:** Optimized for fact table joins
- **Comprehensive Attributes:** 30+ analytics attributes
- **HIPAA Compliance:** De-identification and privacy controls

---

### 8. fact_claims

**Type:** MATERIALIZED_VIEW  
**Row Count:** 413,472  
**Purpose:** Claims fact table with patient dimension joins and comprehensive financial metrics  
**Location:** `s3://databricks-e2demofieldengwest/.../fact_claims`  
**Dimensional Model:** claims_financials  
**Last Refreshed:** 2025-11-22T03:37:31Z  
**Total Size:** 9,088,956 bytes

#### Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `claim_surrogate_key` | string | Claim surrogate key |
| `claim_natural_key` | string | Claim natural key |
| `patient_surrogate_key` | bigint | Patient SK (FK to dim_patients) |
| `claim_year` | int | Claim year |
| `claim_month` | int | Claim month |
| `claim_quarter` | int | Claim quarter |
| `year_month` | string | Year-month identifier |
| `claim_date` | string | Claim date |
| `claim_amount` | double | Claim amount |
| `risk_adjusted_amount` | double | Risk-adjusted amount |
| `high_cost_claim_indicator` | boolean | High-cost claim flag |
| `claim_approved` | boolean | Approval flag |
| `claim_denied` | boolean | Denial flag |
| `claim_paid` | boolean | Payment flag |
| `total_processing_days` | int | Processing duration |
| `claim_processing_efficiency` | string | Processing efficiency category |
| `diagnosis_code` | string | Diagnosis code |
| `procedure_code` | string | Procedure code |
| `medical_coding_complete` | boolean | Coding completeness flag |
| `patient_health_risk_category` | string | Patient health risk category |
| `patient_age_category` | string | Patient age category |
| `patient_region` | string | Patient region |
| `patient_smoking_status` | string | Patient smoking status |
| `patient_premium_category` | string | Patient premium category |
| `claim_data_quality_score` | double | Claim data quality score |
| `combined_data_quality_score` | double | Combined quality score |
| `_pipeline_env` | string | Pipeline environment |
| `fact_last_updated` | timestamp | Last update timestamp |

#### Statistical Properties

| Metric | Value |
|--------|-------|
| **Total Rows** | 413,472 |
| **Distinct Patients** | 40,000 |
| **Distinct Claims** | 25,842 |
| **Distinct Months** | 25 |
| **Claim Amount Statistics** | |
| - Average | $1,218.94 |
| - Minimum | $50.00 |
| - Maximum | $50,000.00 |
| **High-Cost Claims** | 6,816 (1.65%) |
| **Average Processing Days** | 376.21 |

#### Features
- **Dimensional Integration:** Pre-joined with patient dimension
- **Temporal Dimensions:** Year, month, quarter for time-based analysis
- **Financial Metrics:** Risk-adjusted amounts and efficiency metrics
- **Quality Scores:** Combined data quality tracking

---

### 9. fact_medical_events

**Type:** MATERIALIZED_VIEW  
**Row Count:** 1,219,104  
**Purpose:** Medical events fact table with provider analytics and care coordination metrics  
**Location:** `s3://databricks-e2demofieldengwest/.../fact_medical_events`  
**Dimensional Model:** medical_events_clinical  
**Last Refreshed:** 2025-11-22T03:37:44Z  
**Total Size:** 36,023,478 bytes

#### Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `event_surrogate_key` | string | Event surrogate key |
| `event_natural_key` | string | Event natural key |
| `patient_surrogate_key` | bigint | Patient SK (FK to dim_patients) |
| `patient_natural_key` | string | Patient natural key |
| `event_year` | int | Event year |
| `event_month` | int | Event month |
| `event_quarter` | int | Event quarter |
| `event_date` | string | Event date |
| `total_medical_events` | int | Total events count |
| `emergency_visit` | boolean | Emergency visit flag |
| `hospital_admission` | boolean | Hospital admission flag |
| `preventive_care_indicator` | boolean | Preventive care flag |
| `acute_care_indicator` | boolean | Acute care flag |
| `medical_provider` | string | Healthcare provider |
| `provider_type` | string | Provider type |
| `facility_type` | string | Facility type |
| `clinical_outcome_score` | double | Clinical outcome score |
| `care_efficiency_score` | double | Care efficiency score |
| `care_appropriateness_score` | double | Care appropriateness score |
| `visit_duration_minutes` | int | Visit duration |
| `follow_up_required` | boolean | Follow-up required flag |
| `chronic_management_indicator` | boolean | Chronic management flag |
| `is_new_provider` | boolean | New provider flag |
| `provider_patient_familiarity` | double | Provider-patient familiarity |
| `days_since_previous_event` | int | Days since previous event |
| `days_to_next_event` | int | Days to next event |
| `patient_health_risk_category` | string | Patient health risk category |
| `patient_age_category` | string | Patient age category |
| `patient_region` | string | Patient region |
| `demographic_segment` | string | Demographic segment |
| `event_data_quality_score` | double | Event data quality score |
| `_pipeline_env` | string | Pipeline environment |
| `fact_last_updated` | timestamp | Last update timestamp |

#### Statistical Properties

| Metric | Value |
|--------|-------|
| **Total Rows** | 1,219,104 |
| **Distinct Patients** | 40,000 |
| **Distinct Events** | 76,194 |
| **Total Events** | 1,219,104 |
| **Event Categories** | |
| - Emergency Visits | 80,336 (6.59%) |
| - Hospital Admissions | 121,040 (9.93%) |
| **Clinical Scores** | |
| - Average Clinical Outcome Score | 29.98 |
| **Temporal Coverage** | |
| - Distinct Years | 3 |
| - Distinct Months | 12 |

#### Features
- **Provider Analytics:** Provider type and facility classifications
- **Care Coordination:** Provider-patient familiarity metrics
- **Clinical Metrics:** Outcome, efficiency, and appropriateness scores
- **Temporal Analysis:** Event sequencing and timing

---

### 10. fact_claims_monthly_summary

**Type:** MATERIALIZED_VIEW  
**Row Count:** 100  
**Purpose:** Monthly aggregated claims metrics for financial performance trending  
**Location:** `s3://databricks-e2demofieldengwest/.../fact_claims_monthly_summary`  
**Dimensional Model:** claims_monthly_trends  
**Last Refreshed:** 2025-11-22T03:37:40Z  
**Total Size:** 11,540 bytes

#### Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `year_month` | string | Year-month identifier (YYYY-MM) |
| `claim_quarter` | int | Claim quarter |
| `patient_region` | string | Patient region |
| `total_claims` | bigint | Total claims count |
| `unique_patients` | bigint | Unique patients count |
| `total_claim_amount` | double | Total claim amount |
| `avg_claim_amount` | double | Average claim amount |
| `max_claim_amount` | double | Maximum claim amount |
| `min_claim_amount` | double | Minimum claim amount |
| `approved_claims` | bigint | Approved claims count |
| `denied_claims` | bigint | Denied claims count |
| `paid_claims` | bigint | Paid claims count |
| `avg_processing_days` | double | Average processing days |
| `avg_data_quality_score` | double | Average data quality score |
| `high_cost_claims` | bigint | High-cost claims count |
| `high_cost_amount` | double | High-cost amount total |
| `avg_claims_per_patient` | double | Average claims per patient |
| `avg_cost_per_patient` | double | Average cost per patient |

#### Statistical Properties

| Metric | Value |
|--------|-------|
| **Total Rows** | 100 |
| **Distinct Months** | 25 |
| **Distinct Regions** | 4 |
| **Aggregate Metrics** | |
| - Total Claims | 413,472 |
| - Overall Average Claim Amount | $1,222.46 |
| - Total Approved | 330,208 |
| - Total Denied | 42,960 |
| - Overall Average Processing Days | 372.13 |

#### Features
- **Pre-Aggregated:** Optimized for dashboard queries
- **Multi-Dimensional:** Region and time-based aggregations
- **Financial Metrics:** Comprehensive claim statistics
- **Performance Metrics:** Processing efficiency tracking

---

## ML Layer Tables

### 11. ml_insurance_features

**Type:** MANAGED (Delta)  
**Row Count:** 10,000  
**Purpose:** Healthcare-specific features for insurance risk prediction using new schema  
**Location:** `s3://databricks-e2demofieldengwest/.../ml_insurance_features`  
**Primary Key:** `customer_id`  
**Created:** 2025-11-03 13:50:16 UTC

#### Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `patient_surrogate_key` | bigint | Patient surrogate key |
| `patient_natural_key` | string | Patient natural key |
| `patient_id_hash` | string | Hashed patient ID |
| `patient_name_hash` | string | Hashed patient name |
| `effective_date` | timestamp | SCD effective date |
| `expiry_date` | timestamp | SCD expiry date |
| `is_current_record` | boolean | Current record flag |
| `record_version` | int | Record version |
| `patient_age_category` | string | Age category |
| `patient_gender` | string | Patient gender |
| `patient_region` | string | Geographic region |
| `demographic_segment` | string | Demographic segment |
| `health_risk_category` | string | Health risk category |
| `health_risk_score` | double | Health risk score |
| `patient_smoking_status` | string | Smoking status |
| `bmi` | double | Body Mass Index |
| `obesity_indicator` | boolean | Obesity indicator |
| `underweight_indicator` | boolean | Underweight indicator |
| `lifestyle_risk_factors` | string | Lifestyle risk factors |
| `number_of_dependents` | int | Number of dependents |
| `has_dependents` | boolean | Has dependents flag |
| `family_size_category` | string | Family size category |
| `primary_insurance_plan` | string | Primary insurance plan |
| `coverage_effective_date` | string | Coverage effective date |
| `base_premium` | double | Base premium |
| `estimated_annual_premium` | double | Estimated annual premium |
| `patient_premium_category` | string | Premium category |
| `patient_data_quality_score` | double | Data quality score |
| `data_quality_tier` | string | Data quality tier |
| `patient_record_quality` | string | Record quality |
| `hipaa_deidentification_applied` | boolean | HIPAA de-identification flag |
| `age_privacy_protection` | boolean | Age privacy flag |
| `geographic_privacy_protection` | boolean | Geographic privacy flag |
| `data_retention_compliance` | boolean | Data retention flag |
| `_pipeline_env` | string | Pipeline environment |
| `dimension_last_updated` | timestamp | Last update timestamp |
| `age` | int | Patient age (mapped) |
| `children` | int | Number of children (mapped) |
| `smoker` | boolean | Smoking status (mapped) |
| `region` | string | Region (mapped) |
| `sex` | string | Sex (mapped) |
| `age_risk_score` | int | Engineered: Age risk score (3-5) |
| `smoking_impact` | decimal(13,1) | Engineered: Smoking impact factor |
| `family_size_factor` | decimal(14,2) | Engineered: Family size factor |
| `regional_multiplier` | decimal(3,2) | Engineered: Regional multiplier |
| `health_risk_composite` | double | Engineered: Composite health risk |
| `data_quality_score` | double | Engineered: Data quality score |
| `hipaa_compliant` | boolean | HIPAA compliance flag |
| `customer_id` | string | **PRIMARY KEY** - Customer identifier |

#### Statistical Properties

| Metric | Value |
|--------|-------|
| **Total Rows** | 10,000 |
| **Distinct Customers** | 10,000 (100% unique) |
| **Engineered Features** | |
| - Average Age Risk Score | 4.15 |
| - Range | 3-5 |
| - Average Smoking Impact | 69.66 |
| - Average Family Size Factor | 1.18 |
| - Average Regional Multiplier | 1.06 |
| - Average Health Risk Composite | 53.14 |
| - Average Data Quality Score | 1.0 |
| **Categorical Features** | |
| - Distinct Sex Values | 2 |
| - Distinct Regions | 4 |

#### Model Feature Expectations

The trained model expects exactly these 8 features:
1. `age_risk_score` (int)
2. `smoking_impact` (decimal)
3. `family_size_factor` (decimal)
4. `health_risk_composite` (double)
5. `regional_multiplier` (decimal)
6. `data_quality_score` (double)
7. `sex` (string → LabelEncoded)
8. `region` (string → LabelEncoded)

#### Features
- **Feature Store:** Databricks Feature Engineering Client integration
- **Primary Key Constraint:** Enforced on `customer_id`
- **Append-Only:** Supports feature append operations
- **Deletion Vectors:** Enabled for efficient updates
- **Predictive Optimization:** Enabled

---

### 12. ml_patient_predictions

**Type:** MANAGED (Delta)  
**Row Count:** 40,000  
**Purpose:** Model predictions and inference results for patient risk assessment  
**Location:** `s3://databricks-e2demofieldengwest/.../ml_patient_predictions`  
**Created:** 2025-11-03 14:24:51 UTC

#### Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `patient_surrogate_key` | bigint | Patient surrogate key |
| `patient_natural_key` | string | Patient natural key |
| `patient_id_hash` | string | Hashed patient ID |
| `patient_name_hash` | string | Hashed patient name |
| `effective_date` | timestamp | SCD effective date |
| `expiry_date` | timestamp | SCD expiry date |
| `is_current_record` | boolean | Current record flag |
| `record_version` | int | Record version |
| `patient_age_category` | string | Age category |
| `patient_gender` | string | Patient gender |
| `patient_region` | string | Geographic region |
| `demographic_segment` | string | Demographic segment |
| `health_risk_category` | string | Health risk category |
| `health_risk_score` | double | Health risk score |
| `patient_smoking_status` | string | Smoking status |
| `bmi` | double | Body Mass Index |
| `obesity_indicator` | boolean | Obesity indicator |
| `underweight_indicator` | boolean | Underweight indicator |
| `lifestyle_risk_factors` | string | Lifestyle risk factors |
| `number_of_dependents` | int | Number of dependents |
| `has_dependents` | boolean | Has dependents flag |
| `family_size_category` | string | Family size category |
| `primary_insurance_plan` | string | Primary insurance plan |
| `coverage_effective_date` | string | Coverage effective date |
| `base_premium` | double | Base premium |
| `estimated_annual_premium` | double | Estimated annual premium |
| `patient_premium_category` | string | Premium category |
| `patient_data_quality_score` | double | Data quality score |
| `data_quality_tier` | string | Data quality tier |
| `patient_record_quality` | string | Record quality |
| `hipaa_deidentification_applied` | boolean | HIPAA de-identification flag |
| `age_privacy_protection` | boolean | Age privacy flag |
| `geographic_privacy_protection` | boolean | Geographic privacy flag |
| `data_retention_compliance` | boolean | Data retention flag |
| `_pipeline_env` | string | Pipeline environment |
| `dimension_last_updated` | timestamp | Last update timestamp |
| `customer_id` | string | Customer identifier |
| `age_risk_score` | int | Age risk score |
| `smoking_impact` | decimal(13,1) | Smoking impact factor |
| `family_size_factor` | decimal(14,2) | Family size factor |
| `regional_multiplier` | decimal(3,2) | Regional multiplier |
| `health_risk_composite` | double | Composite health risk |
| `data_quality_score` | double | Data quality score |
| `sex` | string | Sex |
| `region` | string | Region |
| `prediction` | double | Raw model prediction |
| `prediction_timestamp` | timestamp | Prediction timestamp |
| `model_version` | string | Model version |
| `model_name` | string | Model name |
| `adjusted_prediction` | double | Adjusted prediction (risk score 0-100) |
| `risk_category` | string | Risk category (low/medium/high/critical) |
| `prediction_lower_bound` | double | Prediction lower bound |
| `prediction_upper_bound` | double | Prediction upper bound |
| `high_risk_patient` | boolean | High-risk patient flag |
| `requires_review` | boolean | Requires review flag |

#### Statistical Properties

| Metric | Value |
|--------|-------|
| **Total Rows** | 40,000 |
| **Distinct Customers** | 10,000 |
| **Prediction Statistics** | |
| - Average Prediction | 53.14 |
| - Minimum | 25.0 |
| - Maximum | 85.0 |
| **Model Information** | |
| - Distinct Model Versions | 1 |
| **Risk Categories** | |
| - High | 17,944 (44.86%) |
| - Low | 13,240 (33.10%) |
| - Medium | 5,220 (13.05%) |
| - Critical | 3,596 (8.99%) |
| **High-Risk Patients** | 9,508 (23.77%) |
| **Requires Review** | 0 (0%) |
| **Temporal Range** | |
| - Earliest Prediction | 2025-11-22 14:55:05 UTC |
| - Latest Prediction | 2025-11-22 14:55:05 UTC |

#### Features
- **Model Tracking:** Model version and name tracking
- **Prediction Intervals:** Upper and lower bounds for uncertainty
- **Risk Categorization:** Business logic for risk assessment
- **Review Flags:** Automated review triggers
- **Deletion Vectors:** Enabled for efficient updates
- **Predictive Optimization:** Enabled

---

### 13. ml_monitoring_summary

**Type:** VIEW  
**Row Count:** 1 (dynamic)  
**Purpose:** Monitoring summary view for last 7 days of predictions  
**Created:** 2025-11-22 06:04:18 UTC

#### Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `period` | string | Time period (e.g., "Last 7 Days") |
| `total_predictions` | bigint | Total predictions count |
| `active_days` | bigint | Distinct days with predictions |
| `avg_risk_score` | double | Average risk score |
| `min_risk_score` | double | Minimum risk score |
| `max_risk_score` | double | Maximum risk score |
| `high_risk_count` | bigint | High-risk predictions count |
| `medium_risk_count` | bigint | Medium-risk predictions count |
| `low_risk_count` | bigint | Low-risk predictions count |
| `generated_at` | timestamp | View generation timestamp |

#### View Definition

```sql
SELECT 
    'Last 7 Days' as period,
    COUNT(*) as total_predictions,
    COUNT(DISTINCT DATE(prediction_timestamp)) as active_days,
    AVG(adjusted_prediction) as avg_risk_score,
    MIN(adjusted_prediction) as min_risk_score,
    MAX(adjusted_prediction) as max_risk_score,
    COUNT(CASE WHEN adjusted_prediction > 80 THEN 1 END) as high_risk_count,
    COUNT(CASE WHEN adjusted_prediction BETWEEN 50 AND 80 THEN 1 END) as medium_risk_count,
    COUNT(CASE WHEN adjusted_prediction < 50 THEN 1 END) as low_risk_count,
    CURRENT_TIMESTAMP() as generated_at
FROM juan_dev.healthcare_data.ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
```

#### Features
- **Dynamic View:** Real-time aggregation from predictions table
- **7-Day Window:** Rolling 7-day monitoring period
- **Risk Distribution:** Counts by risk category
- **Auto-Generated:** Timestamp for view freshness tracking

---

### 14. _quality_monitoring_summary

**Type:** MANAGED (Delta)  
**Row Count:** 696  
**Purpose:** System table for quality monitoring with freshness, completeness, and downstream impact tracking  
**Location:** `s3://databricks-e2demofieldengwest/.../_quality_monitoring_summary`  
**Created:** 2025-09-24 21:26:25 UTC

#### Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `event_time` | timestamp | Event timestamp |
| `catalog_name` | string | Catalog name |
| `schema_name` | string | Schema name |
| `table_name` | string | Table name |
| `catalog_id` | string | Catalog ID |
| `schema_id` | string | Schema ID |
| `table_id` | string | Table ID |
| `status` | string | Overall status |
| `freshness` | struct | Freshness metrics (status, commit_freshness) |
| `completeness` | struct | Completeness metrics (status, total_row_count, daily_row_count) |
| `downstream_impact` | struct | Downstream impact analysis (impact_level, num_downstream_tables, num_queries) |
| `root_cause_analysis` | struct | Root cause analysis (upstream_jobs) |
| `_internal_debug_info` | struct | Internal debugging information |

#### Features
- **Quality Monitoring:** Automated freshness and completeness tracking
- **Downstream Impact:** Analysis of affected downstream tables
- **Root Cause Analysis:** Upstream job tracking
- **Clustered:** Clustered on `schema_id` for performance
- **Row Tracking:** Enabled for audit trails

---

## Data Quality Summary

### Overall Data Quality Metrics

| Table | Row Count | Data Quality Score | Null Rate | Completeness |
|-------|-----------|-------------------|-----------|--------------|
| `silver_patients` | 40,000 | 1.0 | 0% | 100% |
| `silver_claims` | 103,368 | 1.0 | 0% | 100% |
| `silver_medical_events` | 338,748 | 1.0 | 0% | 100% |
| `dim_patients` | 40,000 | 1.0 | 0% | 100% |
| `ml_insurance_features` | 10,000 | 1.0 | 0% | 100% |

### Data Freshness

All materialized views are refreshed via pipeline `a209ea4a-f7a7-4dcf-a822-7c211f7a18f8`:
- **Last Refresh:** 2025-11-22T03:37:00Z
- **Refresh Schedule:** MANUAL
- **Refresh Policy:** AUTO (default)

---

## Relationships and Dependencies

### Primary Keys
- `ml_insurance_features.customer_id` → PRIMARY KEY
- `dim_patients.patient_surrogate_key` → Surrogate key for fact tables
- `fact_claims.patient_surrogate_key` → Foreign key to `dim_patients`
- `fact_medical_events.patient_surrogate_key` → Foreign key to `dim_patients`

### Key Relationships
1. **Bronze → Silver:** Raw data cleaned and validated
2. **Silver → Gold:** Dimensional modeling with SCD Type 2
3. **Gold → ML:** Feature engineering and model training
4. **ML → Predictions:** Batch inference results

### Data Flow
```
bronze_patients → silver_patients → dim_patients → ml_insurance_features
bronze_claims → silver_claims → fact_claims → fact_claims_monthly_summary
bronze_medical_events → silver_medical_events → fact_medical_events
ml_insurance_features → ml_patient_predictions → ml_monitoring_summary
```

---

## Compliance and Governance

### HIPAA Compliance
- **De-identification:** Applied to all patient tables
- **PII Fields:** Tagged in table properties
- **Privacy Controls:** Age and geographic privacy protection
- **Data Retention:** Compliance flags on all tables

### Data Classification
- **PHI:** `bronze_patients`, `silver_patients`
- **PHI_DEIDENTIFIED:** `silver_patients` (after processing)
- **Financial:** `bronze_claims`, `silver_claims`, `fact_claims`
- **Clinical:** `bronze_medical_events`, `silver_medical_events`, `fact_medical_events`
- **Analytics:** `dim_patients`, fact tables

### Table Properties
- **Quality Tiers:** bronze, silver, gold, ML
- **Compliance Tags:** HIPAA, SOX
- **Pipeline Tracking:** All tables tagged with pipeline ID
- **Change Data Feed:** Enabled on key tables

---

## Usage Recommendations

### For ML Training
- Use `ml_insurance_features` for feature lookup
- Join with `dim_patients` for additional attributes
- Filter `dim_patients` with `is_current_record = True`

### For Analytics
- Use `fact_claims` and `fact_medical_events` for dimensional analysis
- Join with `dim_patients` using `patient_surrogate_key`
- Use `fact_claims_monthly_summary` for pre-aggregated metrics

### For Monitoring
- Query `ml_patient_predictions` for prediction history
- Use `ml_monitoring_summary` for quick dashboard metrics
- Check `_quality_monitoring_summary` for data quality issues

### For Data Quality
- All silver and gold tables have `*_data_quality_score` columns
- Quality scores range from 0.0 to 1.0 (1.0 = perfect)
- Current average quality score: 1.0 across all tables

---

## Maintenance Notes

### Refresh Schedule
- **Current:** MANUAL
- **Recommended:** Schedule regular refreshes for materialized views
- **Pipeline:** `a209ea4a-f7a7-4dcf-a822-7c211f7a18f8`

### Storage Optimization
- **Predictive Optimization:** Enabled on managed tables
- **Clustering:** `_quality_monitoring_summary` clustered on `schema_id`
- **Compression:** ZSTD compression on Delta tables

### Monitoring
- Monitor `_quality_monitoring_summary` for freshness and completeness issues
- Track `ml_monitoring_summary` for prediction volume and risk distribution
- Review data quality scores in silver/gold tables

---

**Document Generated:** 2025-11-22  
**Data Source:** `juan_dev.healthcare_data` schema  
**Query Tool:** dbrx-admin-mcp (field-eng-west environment)

