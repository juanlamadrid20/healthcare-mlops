# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a healthcare insurance MLOps project that implements machine learning components for predicting healthcare insurance costs using MLOps best practices on Databricks. The project focuses on HIPAA-compliant healthcare analytics with enterprise-grade governance, monitoring, and compliance features. **Note**: This project assumes pre-existing data tables and does not include data ingestion or ETL pipelines.

## Architecture Components

### 1. Data Layer Dependencies
- **External Data Sources**: This project consumes pre-existing silver and dimensional tables but does not contain data ingestion pipelines
- **Silver Tables**: `juan_dev.healthcare_data.silver_patients` - assumed to exist with clean patient data
- **Dimensional Tables**: `juan_dev.healthcare_data.dim_patients` - assumed to exist with enriched demographics
- **Feature Store**: Advanced healthcare risk features stored in Unity Catalog (`juan_dev.healthcare_data.ml_insurance_features`)
- **Unity Catalog Integration**: Centralized governance for data, features, and models

### 2. Model Training & Registry
- **Training Pipeline**: Random Forest and Gradient Boosting models with healthcare-specific metrics
- **Model Registry**: Models stored in Unity Catalog (`juan_dev.healthcare_data.insurance_model` or `juan_dev.ml.healthcare_insurance_model`)
- **Feature Engineering**: Automated feature lookup using Databricks Feature Engineering Client
- **Target Variable**: Can be either `insurance_charges` (dollar amounts) or `health_risk_score` (0-100 scale)

### 3. Governance & Compliance
- **Model Governance**: Healthcare compliance validation with performance thresholds (R² > 0.70, MAE < $15 for risk scores)
- **HIPAA Compliance**: Synthetic data generation, privacy controls, and compliance tags
- **Automated Promotion**: Models promoted with "champion" alias after passing validation

### 4. Batch Inference & Monitoring
- **Batch Scoring**: Feature Engineering Client integration with Spark optimization
- **Model Monitoring**: Comprehensive drift detection, performance alerts, and executive dashboards
- **Risk Categorization**: Business logic for risk assessment (low/medium/high/critical)

## Key Tables and Schemas

### Primary Data Tables
- `juan_dev.healthcare_data.silver_patients` - Clean patient data
- `juan_dev.healthcare_data.dim_patients` - Dimensional patient data with demographics
- `juan_dev.healthcare_data.ml_insurance_features` - Feature store table
- `juan_dev.healthcare_data.ml_patient_predictions` - Batch inference results

### Monitoring Tables  
- `juan_dev.ml.insurance_silver` - Training baseline data
- `juan_dev.ml.insurance_predictions` - Live prediction results for monitoring
- Monitoring views: `healthcare_drift_detection`, `healthcare_model_alerts`, `model_performance_dashboard`

## Development Workflow

### Data Processing
- **Prerequisite**: Ensure `juan_dev.healthcare_data.silver_patients` and `juan_dev.healthcare_data.dim_patients` tables exist before running ML components
- Use `patient_natural_key` as the primary customer identifier across all components
- Always filter `dim_patients` with `is_current_record = True` for active records
- Map categorical fields to numeric values for ML compatibility (see feature engineering notebooks)

### Feature Engineering
- Features are centrally managed through Databricks Feature Engineering Client
- Use `FeatureLookup` for automated feature joining during training and inference
- Key engineered features: `age_risk_score`, `smoking_impact`, `health_risk_composite`, `regional_multiplier`

### Model Training
- Use MLflow for experiment tracking with Unity Catalog registry
- Set registry URI: `mlflow.set_registry_uri("databricks-uc")`
- Models include embedded preprocessing pipelines for consistent inference
- Log models with feature engineering integration using `fe.log_model()`

### Model Governance
- Run governance checks before promoting models to production
- Use `champion` alias for production models
- Validation includes healthcare-specific requirements and bias testing
- Models automatically tagged with compliance metadata

### Batch Inference
- Always use Feature Engineering Client for consistent feature lookup
- Apply business rules and risk categorization post-prediction  
- Include prediction metadata (timestamp, model version, confidence intervals)

### Monitoring
- Set up comprehensive monitoring including drift detection, alerts, and dashboards
- Monitor for demographic bias and healthcare equity metrics
- Use unified schema approach (`juan_dev.ml`) for all monitoring assets
- Run diagnostics before setting up monitoring components

## Healthcare-Specific Considerations

### Compliance Requirements
- All synthetic data generation must be HIPAA-compliant
- Models require healthcare compliance validation before promotion
- Data quality scores and HIPAA flags tracked throughout pipeline

### Business Metrics
- Performance targets: R² > 0.80, MAE < $3,000 for cost prediction, MAE < 15 for risk scores
- High-risk patient identification using 95th percentile thresholds
- Regional equity monitoring to ensure fair access and pricing

### Clinical Validation
- BMI categorization follows clinical standards (underweight < 18.5, normal 18.5-24.9, overweight 25-29.9, obese ≥30)
- Age risk scoring on 1-5 scale based on healthcare actuarial standards
- Smoking impact factors aligned with health insurance industry practices

## Common Development Commands

### Running Notebooks
```python
# Execute notebooks in order:
# 1. 00-training/00-insurance-model-feature.ipynb (Feature Engineering)
# 2. 00-training/01-insurance-model-train.ipynb (Model Training)  
# 3. 01-governance/insurance-model-governance.py (Model Governance)
# 4. 02-batch/insurance-model-batch.ipynb (Batch Inference)
# 5. 03-monitoring/insurance-model-monitor.ipynb (Model Monitoring)
```

### Testing and Validation
This project uses Databricks notebooks rather than traditional Python testing frameworks. Validation is performed through:
- Model governance checks with healthcare-specific thresholds
- Monitoring diagnostics and health checks
- Feature engineering validation through data quality scores

### Data Quality Verification
```sql
-- Verify prerequisite tables exist and have data
SELECT * FROM juan_dev.healthcare_data.silver_patients LIMIT 10;
SELECT * FROM juan_dev.healthcare_data.dim_patients WHERE is_current_record = True LIMIT 10;

-- Check feature store after running feature engineering
SELECT * FROM juan_dev.healthcare_data.ml_insurance_features LIMIT 10;
```

### Model Management
```python
# Check current champion model
client = MlflowClient()
champion_info = client.get_model_version_by_alias(model_name, "champion")

# Run governance on latest model
governance = ModelGovernance()
result = governance.run_governance()
```

### Monitoring System Setup
```python
# Initialize monitoring
monitor = HealthcareModelMonitor()

# Run diagnostics first
diagnostic_results = monitor.diagnose_and_fix_setup_issues()

# Set up complete monitoring if ready
if diagnostic_results["ready_for_setup"]:
    setup_results = monitor.setup_complete_monitoring_system()
```

## File Structure Notes

- `00-training/`: Feature engineering and model training notebooks
- `01-governance/`: Model governance and validation scripts  
- `02-batch/`: Batch inference pipelines
- `03-monitoring/`: Model monitoring and observability
- `99-eda/`: Exploratory data analysis and SQL queries

The project follows a numbered workflow pattern where components should generally be executed in sequential order for proper dependency management.