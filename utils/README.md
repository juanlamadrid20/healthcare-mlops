# Healthcare MLOps - Validation Utilities

This directory contains comprehensive business validation tools for the healthcare insurance risk prediction model.

## Overview

The validation framework validates the **business value and clinical relevance** of the model, going beyond technical correctness to ensure the model delivers on its business objectives.

## Validation Assets

### 1. Updated Existing Notebooks

#### [data_validation.ipynb](./data_validation.ipynb)
**Purpose**: Validate data prerequisites before running batch inference

**Key Updates**:
- Fixed column names to match actual schema (patient_gender, bmi numeric)
- Added data quality checks (null rates, BMI ranges)
- Validates required columns exist

**Usage**:
```python
dbutils.notebook.run(
    "./utils/data_validation",
    timeout_seconds=300,
    arguments={
        "catalog": "juan_dev",
        "schema": "healthcare_data",
        "validation_date": "2025-11-03"
    }
)
```

#### [inference_validation.ipynb](./inference_validation.ipynb)
**Purpose**: Validate prediction quality and business logic

**Key Updates**:
- Business KPI thresholds (5-30% high-risk, 30%+ low-risk, <10% critical)
- Clinical relevance checks (smoking/age correlation)
- Alert rules for unusual distributions

**Usage**:
```python
dbutils.notebook.run(
    "./utils/inference_validation",
    timeout_seconds=300,
    arguments={
        "catalog": "juan_dev",
        "ml_schema": "healthcare_data",
        "predictions_table": "ml_patient_predictions",
        "batch_date": "2025-11-03"
    }
)
```

---

### 2. New Business Validation Assets

#### [business_validation.ipynb](./business_validation.ipynb) ⭐
**Purpose**: Comprehensive business KPI and clinical validation

**Validation Areas**:
1. **Business KPIs**: Volume, high-risk %, average risk scores
2. **Clinical Relevance**: Smoking impact, age correlation, BMI relationships
3. **Regional Equity**: Fair predictions across regions (<20% disparity)
4. **Prediction Stability**: Temporal consistency (CV < 10%)
5. **ROI Analysis**: Intervention costs vs prevented claims value
6. **Model Governance**: Champion model status and compliance tags

**Key Metrics**:
- High-risk %: Target 5-25%
- Average risk: Target 20-60
- Regional disparity: < 20%
- Prediction stability: CV < 10%
- ROI: > 50% acceptable, > 100% excellent

**Usage**:
```python
dbutils.notebook.run(
    "./utils/business_validation",
    timeout_seconds=600,
    arguments={
        "catalog": "juan_dev",
        "ml_schema": "healthcare_data",
        "model_name": "insurance_model",
        "lookback_days": "30"
    }
)
```

**Returns**: JSON with validation summary
```json
{
  "status": "EXCELLENT|GOOD|ACCEPTABLE|NEEDS_IMPROVEMENT",
  "total_checks": 15,
  "passed": 13,
  "warnings": 2,
  "failures": 0,
  "total_predictions": 10000
}
```

---

#### [sql/validation_queries.sql](./sql/validation_queries.sql)
**Purpose**: Reusable SQL queries for ad-hoc analysis and monitoring

**12 Query Collections**:
1. Daily Prediction Summary
2. Risk Category Distribution
3. Risk by Demographics (smoking, age, BMI, gender)
4. Regional Equity Analysis
5. High-Risk Patient Identification
6. Model Performance Tracking
7. Data Quality Monitoring
8. Business Rule Compliance
9. Comparative Analysis (current vs historical)
10. Actionable Insights (intervention candidates)
11. Executive Dashboard Metrics
12. Alerts and Anomalies

**Usage**:
- Copy queries to Databricks SQL Editor
- Replace variables: `${catalog}`, `${ml_schema}`
- Run interactively or schedule as queries

**Example**:
```sql
-- Daily prediction summary
SELECT
  DATE(prediction_timestamp) as prediction_date,
  COUNT(*) as total_predictions,
  AVG(adjusted_prediction) as avg_risk_score,
  SUM(CASE WHEN high_risk_patient = true THEN 1 ELSE 0 END) as high_risk_count
FROM juan_dev.healthcare_data.ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 30 DAYS
GROUP BY DATE(prediction_timestamp)
ORDER BY prediction_date DESC;
```

---

#### [dashboard_metrics.py](./dashboard_metrics.py)
**Purpose**: Generate metrics for executive dashboards and monitoring

**Generated Metrics**:
- Volume: Predictions, unique patients, daily averages
- Risk: Distribution, averages, high-risk %
- Demographics: By smoking, age, gender
- Regional: Equity analysis with disparity %
- Quality: Null rates, outliers, confidence intervals
- Model: Champion version, R², MAE, accuracy
- Trends: Period-over-period changes
- Business: ROI, intervention costs, prevented claims
- Alerts: Automatic anomaly detection

**Alert Thresholds**:
- High-risk % out of range (5-30%)
- Average risk unusual (20-70)
- High null rate (> 1%)
- Regional disparity (> 25%)

**Output Formats**:
- JSON: For API consumption
- Delta Table: Historical tracking (`ml_dashboard_metrics`)

**Usage**:
```python
# Run as Databricks notebook
dbutils.notebook.run(
    "./utils/dashboard_metrics",
    timeout_seconds=600,
    arguments={
        "catalog": "juan_dev",
        "ml_schema": "healthcare_data",
        "model_name": "insurance_model",
        "lookback_days": "7",
        "output_format": "both"  # json, delta, or both
    }
)
```

---

#### [validation_report.ipynb](./validation_report.ipynb) 📊
**Purpose**: Executive-ready comprehensive validation report

**Report Sections**:
1. **Executive Summary**: KPIs, business impact, ROI
2. **Technical Validation**: Data quality, prediction volume (runs data_validation)
3. **Business Performance**: Risk distribution, intervention candidates
4. **Clinical Relevance**: Medical validity checks
5. **Regional Equity & Fairness**: Geographic and gender analysis
6. **Compliance & Governance**: Model status, HIPAA tags
7. **Recommendations**: Prioritized action items (HIGH/MEDIUM/LOW)
8. **Final Scorecard**: Overall validation score

**Validation Categories**:
- Technical Validation
- Business Performance
- Clinical Relevance
- Regional Equity
- Model Governance

**Overall Assessment**:
- EXCELLENT: 90%+ checks passed
- GOOD: 70-89% checks passed
- ACCEPTABLE: 50-69% checks passed
- NEEDS_IMPROVEMENT: < 50% checks passed

**Usage**:
```python
dbutils.notebook.run(
    "./utils/validation_report",
    timeout_seconds=900,
    arguments={
        "catalog": "juan_dev",
        "ml_schema": "healthcare_data",
        "model_name": "insurance_model",
        "report_period_days": "30"
    }
)
```

**Returns**: JSON summary
```json
{
  "status": "EXCELLENT",
  "overall_score": 93.0,
  "categories_passed": 5,
  "total_categories": 5,
  "high_priority_actions": 0,
  "roi": 234.5,
  "patients_evaluated": 10000,
  "high_risk_identified": 1500
}
```

---

## Recommended Workflow

### For Daily Monitoring
1. Run **dashboard_metrics.py** daily to track KPIs
2. Set up alerts based on metric thresholds
3. Use **SQL queries** for ad-hoc investigation

### For Weekly/Monthly Reviews
1. Run **business_validation.ipynb** for comprehensive validation
2. Review clinical relevance and equity metrics
3. Generate **validation_report.ipynb** for stakeholders

### Before Production Deployment
1. Run **data_validation.ipynb** to ensure data quality
2. Run **inference_validation.ipynb** after batch scoring
3. Run **validation_report.ipynb** for governance approval
4. Verify all checks pass with EXCELLENT or GOOD status

---

## Key Business Validation Criteria

### Business KPIs
- ✅ High-risk identification: 5-25% (prevents under/over-flagging)
- ✅ Average risk: 20-60 (moderate population health)
- ✅ Prediction volume: > 100 (statistical validity)

### Clinical Validity
- ✅ Smoking increases risk (medical knowledge)
- ✅ Age positively correlated with risk (actuarial standards)
- ✅ Obesity increases risk (BMI > 30 higher than normal)

### Equity & Fairness
- ✅ Regional disparity: < 20% (fair across geographies)
- ✅ Gender disparity: < 10 risk points (no gender bias)

### Business Impact
- ✅ ROI > 50%: Acceptable business value
- ✅ ROI > 100%: Excellent business value
- ✅ Net benefit positive (prevented claims > costs)

---

## Important Notes

### Column Name Mappings
The validation notebooks use the actual column names from the predictions table:
- `patient_smoking_status` (not `smoking_status`)
- `patient_age_category` (not `age_category`)
- `prediction_lower_bound` / `prediction_upper_bound` (not `confidence_interval_*`)
- `bmi` is numeric (derived into categories in validation)

### Configurable Assumptions
Business impact calculations use configurable assumptions in **business_validation.ipynb**:
```python
assumptions = {
    "avg_intervention_cost": 500,         # Cost per intervention
    "avg_prevented_claim": 5000,          # Value of prevented claim
    "intervention_success_rate": 0.30,    # 30% success rate
    "model_operating_cost_monthly": 2000  # Monthly model costs
}
```

Adjust these based on your actual business metrics for accurate ROI calculations.

---

## Integration with MLOps Pipeline

The validation framework integrates with your existing MLOps workflow:

```
Training → Governance → Batch Inference → Validation → Production
                ↓            ↓               ↓
              Champion    Predictions    Business Value
              Promotion    Generated      Confirmed
```

- After **model governance** promotes champion model
- After **batch inference** generates predictions
- **Validation** confirms business value before next deployment cycle

---

## Troubleshooting

### Issue: No predictions found
**Solution**: Check the `lookback_days` parameter or verify batch inference has run

### Issue: Column not found errors
**Solution**: Verify column names match the predictions table schema using `DESCRIBE TABLE`

### Issue: Low validation scores
**Solution**: Review specific failed checks in report recommendations section

---

## Next Steps

1. **Run validation notebooks** on existing predictions
2. **Review validation reports** and identify any issues
3. **Schedule dashboard_metrics.py** for daily monitoring
4. **Set up alerts** for critical thresholds
5. **Share validation_report** with stakeholders for governance approval

---

**Last Updated**: 2025-11-03
**Contact**: Data Science Team
