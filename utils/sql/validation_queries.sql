-- ========================================
-- Healthcare Insurance Risk Model
-- Business Validation SQL Queries
-- ========================================
--
-- This collection contains reusable SQL queries for validating
-- the business purpose and results of the healthcare insurance
-- risk prediction model.
--
-- Usage: Replace variables like ${catalog}, ${ml_schema} with actual values
--        or use Databricks widgets/parameters
-- ========================================

-- ========================================
-- 1. DAILY PREDICTION SUMMARY
-- ========================================
-- Purpose: Get daily summary of predictions for monitoring trends

SELECT
  DATE(prediction_timestamp) as prediction_date,
  COUNT(*) as total_predictions,
  AVG(adjusted_prediction) as avg_risk_score,
  STDDEV(adjusted_prediction) as risk_score_std,
  MIN(adjusted_prediction) as min_risk,
  MAX(adjusted_prediction) as max_risk,
  SUM(CASE WHEN high_risk_patient = true THEN 1 ELSE 0 END) as high_risk_count,
  ROUND(SUM(CASE WHEN high_risk_patient = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as high_risk_pct,
  COUNT(DISTINCT customer_id) as unique_patients
FROM ${catalog}.${ml_schema}.ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 30 DAYS
GROUP BY DATE(prediction_timestamp)
ORDER BY prediction_date DESC;

-- ========================================
-- 2. RISK CATEGORY DISTRIBUTION
-- ========================================
-- Purpose: Understand the distribution of risk categories

SELECT
  risk_category,
  COUNT(*) as patient_count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage,
  AVG(adjusted_prediction) as avg_risk_score,
  MIN(adjusted_prediction) as min_risk_score,
  MAX(adjusted_prediction) as max_risk_score
FROM ${catalog}.${ml_schema}.ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY risk_category
ORDER BY
  CASE risk_category
    WHEN 'critical' THEN 1
    WHEN 'high' THEN 2
    WHEN 'medium' THEN 3
    WHEN 'low' THEN 4
    ELSE 5
  END;

-- ========================================
-- 3. RISK BY DEMOGRAPHICS
-- ========================================
-- Purpose: Validate clinical relevance and check for demographic bias

-- Risk by Smoking Status
SELECT
  smoking_status,
  COUNT(*) as patient_count,
  AVG(adjusted_prediction) as avg_risk_score,
  STDDEV(adjusted_prediction) as risk_std,
  ROUND(SUM(CASE WHEN high_risk_patient = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as high_risk_pct
FROM ${catalog}.${ml_schema}.ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY smoking_status
ORDER BY avg_risk_score DESC;

-- Risk by Age Category
SELECT
  age_category,
  COUNT(*) as patient_count,
  AVG(adjusted_prediction) as avg_risk_score,
  STDDEV(adjusted_prediction) as risk_std,
  ROUND(SUM(CASE WHEN high_risk_patient = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as high_risk_pct
FROM ${catalog}.${ml_schema}.ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY age_category
ORDER BY age_category;

-- Risk by BMI Category
SELECT
  bmi_category,
  COUNT(*) as patient_count,
  AVG(adjusted_prediction) as avg_risk_score,
  ROUND(SUM(CASE WHEN high_risk_patient = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as high_risk_pct
FROM ${catalog}.${ml_schema}.ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY bmi_category
ORDER BY
  CASE bmi_category
    WHEN 'underweight' THEN 1
    WHEN 'normal' THEN 2
    WHEN 'overweight' THEN 3
    WHEN 'obese' THEN 4
    ELSE 5
  END;

-- Risk by Gender
SELECT
  sex as gender,
  COUNT(*) as patient_count,
  AVG(adjusted_prediction) as avg_risk_score,
  ROUND(SUM(CASE WHEN high_risk_patient = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as high_risk_pct
FROM ${catalog}.${ml_schema}.ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY sex
ORDER BY avg_risk_score DESC;

-- ========================================
-- 4. REGIONAL EQUITY ANALYSIS
-- ========================================
-- Purpose: Ensure fair predictions across geographic regions

SELECT
  region,
  COUNT(*) as patient_count,
  AVG(adjusted_prediction) as avg_risk_score,
  STDDEV(adjusted_prediction) as risk_std,
  MIN(adjusted_prediction) as min_risk,
  MAX(adjusted_prediction) as max_risk,
  ROUND(SUM(CASE WHEN high_risk_patient = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as high_risk_pct,
  AVG(confidence_interval_upper - confidence_interval_lower) as avg_ci_width
FROM ${catalog}.${ml_schema}.ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY region
ORDER BY avg_risk_score DESC;

-- Regional Disparity Check
WITH regional_metrics AS (
  SELECT
    region,
    AVG(adjusted_prediction) as avg_risk
  FROM ${catalog}.${ml_schema}.ml_patient_predictions
  WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
  GROUP BY region
)
SELECT
  MAX(avg_risk) as max_regional_risk,
  MIN(avg_risk) as min_regional_risk,
  MAX(avg_risk) - MIN(avg_risk) as risk_disparity,
  ROUND(((MAX(avg_risk) - MIN(avg_risk)) / MIN(avg_risk)) * 100, 2) as disparity_pct
FROM regional_metrics;

-- ========================================
-- 5. HIGH-RISK PATIENT IDENTIFICATION
-- ========================================
-- Purpose: Validate high-risk patient identification effectiveness

-- High-Risk Patient Summary
SELECT
  COUNT(*) as total_high_risk_patients,
  AVG(adjusted_prediction) as avg_risk_score,
  MIN(adjusted_prediction) as min_risk_score,
  COUNT(DISTINCT region) as regions_represented,
  ROUND(SUM(CASE WHEN smoking_status = 'yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as smoker_pct,
  ROUND(SUM(CASE WHEN bmi_category = 'obese' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as obese_pct
FROM ${catalog}.${ml_schema}.ml_patient_predictions
WHERE high_risk_patient = true
  AND prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS;

-- High-Risk by Risk Category
SELECT
  risk_category,
  COUNT(*) as patient_count,
  AVG(adjusted_prediction) as avg_risk_score,
  AVG(number_of_dependents) as avg_dependents
FROM ${catalog}.${ml_schema}.ml_patient_predictions
WHERE high_risk_patient = true
  AND prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY risk_category
ORDER BY avg_risk_score DESC;

-- ========================================
-- 6. MODEL PERFORMANCE TRACKING
-- ========================================
-- Purpose: Track model health and prediction quality over time

-- Weekly Performance Trends
SELECT
  DATE_TRUNC('week', prediction_timestamp) as week_start,
  COUNT(*) as prediction_count,
  AVG(adjusted_prediction) as avg_risk_score,
  STDDEV(adjusted_prediction) as risk_std,
  ROUND(SUM(CASE WHEN high_risk_patient = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as high_risk_pct,
  AVG(confidence_interval_upper - confidence_interval_lower) as avg_ci_width,
  MIN(prediction_timestamp) as earliest_prediction,
  MAX(prediction_timestamp) as latest_prediction
FROM ${catalog}.${ml_schema}.ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 90 DAYS
GROUP BY DATE_TRUNC('week', prediction_timestamp)
ORDER BY week_start DESC;

-- Prediction Quality Metrics
SELECT
  COUNT(*) as total_predictions,
  SUM(CASE WHEN adjusted_prediction IS NULL THEN 1 ELSE 0 END) as null_predictions,
  SUM(CASE WHEN adjusted_prediction < 0 THEN 1 ELSE 0 END) as negative_predictions,
  SUM(CASE WHEN adjusted_prediction > 100 THEN 1 ELSE 0 END) as outlier_predictions,
  AVG(confidence_interval_upper - confidence_interval_lower) as avg_uncertainty,
  ROUND(SUM(CASE WHEN adjusted_prediction IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as prediction_success_rate
FROM ${catalog}.${ml_schema}.ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS;

-- ========================================
-- 7. DATA QUALITY MONITORING
-- ========================================
-- Purpose: Monitor data quality and completeness

-- Feature Completeness Check
SELECT
  COUNT(*) as total_records,
  SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) as missing_customer_id,
  SUM(CASE WHEN smoking_status IS NULL THEN 1 ELSE 0 END) as missing_smoking_status,
  SUM(CASE WHEN region IS NULL THEN 1 ELSE 0 END) as missing_region,
  SUM(CASE WHEN age_category IS NULL THEN 1 ELSE 0 END) as missing_age_category,
  SUM(CASE WHEN bmi_category IS NULL THEN 1 ELSE 0 END) as missing_bmi_category,
  SUM(CASE WHEN sex IS NULL THEN 1 ELSE 0 END) as missing_sex
FROM ${catalog}.${ml_schema}.ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS;

-- Duplicate Prediction Check
SELECT
  customer_id,
  DATE(prediction_timestamp) as prediction_date,
  COUNT(*) as prediction_count
FROM ${catalog}.${ml_schema}.ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY customer_id, DATE(prediction_timestamp)
HAVING COUNT(*) > 1
ORDER BY prediction_count DESC
LIMIT 100;

-- ========================================
-- 8. BUSINESS RULE COMPLIANCE
-- ========================================
-- Purpose: Validate that predictions follow business logic rules

-- Verify High-Risk Flag Logic (should be > 95th percentile)
WITH risk_percentile AS (
  SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY adjusted_prediction) as p95_threshold
  FROM ${catalog}.${ml_schema}.ml_patient_predictions
  WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
)
SELECT
  COUNT(*) as total_high_risk_flagged,
  SUM(CASE WHEN adjusted_prediction >= p95_threshold THEN 1 ELSE 0 END) as correctly_flagged,
  SUM(CASE WHEN adjusted_prediction < p95_threshold THEN 1 ELSE 0 END) as incorrectly_flagged,
  ROUND(SUM(CASE WHEN adjusted_prediction >= p95_threshold THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as flag_accuracy_pct
FROM ${catalog}.${ml_schema}.ml_patient_predictions, risk_percentile
WHERE high_risk_patient = true
  AND prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS;

-- Verify Risk Category Boundaries
SELECT
  risk_category,
  MIN(adjusted_prediction) as min_risk,
  MAX(adjusted_prediction) as max_risk,
  COUNT(*) as patient_count
FROM ${catalog}.${ml_schema}.ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY risk_category
ORDER BY min_risk;

-- ========================================
-- 9. COMPARATIVE ANALYSIS
-- ========================================
-- Purpose: Compare current predictions to historical baselines

-- Current vs Last Week Comparison
WITH current_week AS (
  SELECT
    AVG(adjusted_prediction) as avg_risk,
    COUNT(*) as prediction_count,
    SUM(CASE WHEN high_risk_patient = true THEN 1 ELSE 0 END) as high_risk_count
  FROM ${catalog}.${ml_schema}.ml_patient_predictions
  WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
),
last_week AS (
  SELECT
    AVG(adjusted_prediction) as avg_risk,
    COUNT(*) as prediction_count,
    SUM(CASE WHEN high_risk_patient = true THEN 1 ELSE 0 END) as high_risk_count
  FROM ${catalog}.${ml_schema}.ml_patient_predictions
  WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 14 DAYS
    AND prediction_timestamp < CURRENT_DATE() - INTERVAL 7 DAYS
)
SELECT
  current_week.avg_risk as current_avg_risk,
  last_week.avg_risk as last_week_avg_risk,
  ROUND(current_week.avg_risk - last_week.avg_risk, 2) as risk_change,
  ROUND(((current_week.avg_risk - last_week.avg_risk) / last_week.avg_risk) * 100, 2) as risk_change_pct,
  current_week.prediction_count as current_volume,
  last_week.prediction_count as last_week_volume,
  current_week.prediction_count - last_week.prediction_count as volume_change,
  ROUND((current_week.high_risk_count * 100.0 / current_week.prediction_count), 2) as current_high_risk_pct,
  ROUND((last_week.high_risk_count * 100.0 / last_week.prediction_count), 2) as last_week_high_risk_pct
FROM current_week, last_week;

-- ========================================
-- 10. ACTIONABLE INSIGHTS
-- ========================================
-- Purpose: Identify patients requiring immediate intervention

-- Top High-Risk Patients Requiring Intervention
SELECT
  customer_id,
  adjusted_prediction as risk_score,
  risk_category,
  smoking_status,
  bmi_category,
  age_category,
  region,
  number_of_dependents,
  confidence_interval_lower,
  confidence_interval_upper,
  prediction_timestamp
FROM ${catalog}.${ml_schema}.ml_patient_predictions
WHERE high_risk_patient = true
  AND prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
ORDER BY adjusted_prediction DESC
LIMIT 100;

-- Patients with Rapidly Increasing Risk (requires historical predictions)
WITH patient_risk_trend AS (
  SELECT
    customer_id,
    DATE(prediction_timestamp) as prediction_date,
    AVG(adjusted_prediction) as daily_avg_risk,
    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY DATE(prediction_timestamp) DESC) as recency_rank
  FROM ${catalog}.${ml_schema}.ml_patient_predictions
  WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 30 DAYS
  GROUP BY customer_id, DATE(prediction_timestamp)
),
risk_comparison AS (
  SELECT
    customer_id,
    MAX(CASE WHEN recency_rank = 1 THEN daily_avg_risk END) as latest_risk,
    MAX(CASE WHEN recency_rank = 7 THEN daily_avg_risk END) as week_ago_risk
  FROM patient_risk_trend
  WHERE recency_rank <= 7
  GROUP BY customer_id
  HAVING MAX(CASE WHEN recency_rank = 1 THEN daily_avg_risk END) IS NOT NULL
    AND MAX(CASE WHEN recency_rank = 7 THEN daily_avg_risk END) IS NOT NULL
)
SELECT
  customer_id,
  latest_risk,
  week_ago_risk,
  ROUND(latest_risk - week_ago_risk, 2) as risk_increase,
  ROUND(((latest_risk - week_ago_risk) / week_ago_risk) * 100, 2) as risk_increase_pct
FROM risk_comparison
WHERE latest_risk > week_ago_risk
ORDER BY risk_increase DESC
LIMIT 50;

-- ========================================
-- 11. EXECUTIVE DASHBOARD METRICS
-- ========================================
-- Purpose: Key metrics for executive dashboards and reports

SELECT
  'Last 7 Days' as time_period,
  COUNT(DISTINCT customer_id) as unique_patients_scored,
  COUNT(*) as total_predictions,
  ROUND(AVG(adjusted_prediction), 2) as avg_risk_score,
  SUM(CASE WHEN high_risk_patient = true THEN 1 ELSE 0 END) as high_risk_patients,
  ROUND(SUM(CASE WHEN high_risk_patient = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as high_risk_pct,
  ROUND(SUM(CASE WHEN risk_category = 'critical' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as critical_pct,
  ROUND(SUM(CASE WHEN risk_category = 'low' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as low_risk_pct,
  COUNT(DISTINCT region) as regions_covered,
  ROUND(AVG(confidence_interval_upper - confidence_interval_lower), 2) as avg_prediction_uncertainty
FROM ${catalog}.${ml_schema}.ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS;

-- ========================================
-- 12. ALERTS AND ANOMALIES
-- ========================================
-- Purpose: Identify anomalies that may require investigation

-- Unusual Risk Distribution Alert
WITH daily_metrics AS (
  SELECT
    DATE(prediction_timestamp) as prediction_date,
    AVG(adjusted_prediction) as avg_risk,
    STDDEV(adjusted_prediction) as risk_std
  FROM ${catalog}.${ml_schema}.ml_patient_predictions
  WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 30 DAYS
  GROUP BY DATE(prediction_timestamp)
),
baseline AS (
  SELECT
    AVG(avg_risk) as baseline_avg,
    STDDEV(avg_risk) as baseline_std
  FROM daily_metrics
  WHERE prediction_date < CURRENT_DATE() - INTERVAL 7 DAYS
)
SELECT
  dm.prediction_date,
  dm.avg_risk,
  b.baseline_avg,
  ROUND(dm.avg_risk - b.baseline_avg, 2) as deviation,
  ROUND((dm.avg_risk - b.baseline_avg) / b.baseline_std, 2) as z_score,
  CASE
    WHEN ABS((dm.avg_risk - b.baseline_avg) / b.baseline_std) > 2 THEN 'ALERT'
    WHEN ABS((dm.avg_risk - b.baseline_avg) / b.baseline_std) > 1.5 THEN 'WARNING'
    ELSE 'NORMAL'
  END as alert_status
FROM daily_metrics dm, baseline b
WHERE dm.prediction_date >= CURRENT_DATE() - INTERVAL 7 DAYS
ORDER BY dm.prediction_date DESC;

-- ========================================
-- END OF VALIDATION QUERIES
-- ========================================
