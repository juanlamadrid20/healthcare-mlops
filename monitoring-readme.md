# Model Monitoring README

## Overview

This document provides an understanding of how the `insurance-model-monitor.ipynb` notebook is utilized for monitoring model performance using Lakehouse architecture. The notebook focuses on key aspects of model monitoring, tackling challenges such as drift detection, data quality, and alert systems.

## Components

### Initialization

- **SimpleHealthcareModelMonitor**: A central class designed to oversee the monitoring processes. It encapsulates the setup of various views and monitoring configurations.
- **ML Prefixing**: All the monitoring views/tables are prefixed with 'ml_' to distinguish them from business tables. This provides a clear separation and easier management.

### Key Functionalities

1. **Table Access Verification**:
   - Checks both baseline and monitoring tables to ensure they are accessible and valid.
   - Outputs the count and sample data from the tables for a preliminary health check.

2. **Drift Detection**:
   - Establishes a view to identify shifts in predictions over time.
   - Utilizes statistical analyses such as average, minimum, maximum, and standard deviation calculations.

3. **Summary and Alerts**:
   - Creates summary views displaying prediction statistics over a defined timeframe.
   - Sets up alert systems based on volume and risk thresholds.

4. **Integration with Lakehouse Monitoring**:
   - Configures minimal to advanced monitoring setups using Databricks Lakehouse.
   - Incorporates scheduling and alerting mechanisms for real-time monitoring.

## Best Practices Alignment

- **Data Integrity**: Initially verifies table access and data integrity to preemptively catch potential data issues.
- **Continuous Monitoring**: Employs scheduled checks and drift detection to continuously assess model performance and drift.
- **Alerting & Notification**: Serves stakeholders by providing alerts for abnormal conditions such as high-risk score averages or low data volumes.
- **Scalable & Extensible**: Designed to be scalable and easily modifiable to incorporate additional business logic or more complex scenarios.
- **Clear Organization**: The usage of ML prefixes aids in categorizing and accessing views efficiently, maintaining a clean data environment.

## Execution

To set up and validate model monitoring:

1. Initialize the monitoring system by running the defined cells in the notebook.
2. Execute the setup functions to create necessary monitoring components like drift detection and summary views.
3. Verify all views are operational with provided SQL commands.
4. Monitor alerts and logs for real-time insights and take necessary action based on alerts.

## Next Steps

1. Validate the basic setup with initial data.
2. Integrate business-specific monitoring logic incrementally.
3. Ensure that native Lakehouse Monitoring optimization is achieved for scaling monitoring needs.
4. Regularly review and refine monitoring parameters in response to business requirements or model updates.

By following these guidelines and leveraging the capabilities outlined here, you can maintain effective oversight over model performance and swiftly respond to any anomalies.

## Tables Being Monitored

### 1. **Primary Monitoring Table**
- **Table**: `juan_dev.healthcare_data.ml_patient_predictions`
- **Purpose**: Contains real-time model predictions and inference logs
- **Key Columns**:
  - `prediction_timestamp` - When the prediction was made
  - `adjusted_prediction` - The model's risk score output
  - `model_name` - Identifier for the model version
  - `customer_id` - Patient/customer identifier
  - `smoker`, `age`, `bmi` - Input features for demographic tracking

### 2. **Baseline Reference Table**
- **Table**: `juan_dev.healthcare_data.silver_patients`
- **Purpose**: Historical baseline data for drift comparison
- **Use**: Reference point for detecting data drift in input features

## How They're Being Monitored

### **1. Drift Detection Monitoring**
**View**: `juan_dev.healthcare_data.ml_drift_monitor`

```sql
-- Tracks daily prediction patterns and feature drift
SELECT 
    DATE(prediction_timestamp) as prediction_date,
    COUNT(*) as daily_predictions,
    AVG(adjusted_prediction) as avg_prediction,
    MIN/MAX/STDDEV(adjusted_prediction) as prediction_distribution,
    
    -- Feature drift tracking
    AVG(CASE WHEN smoker THEN 1.0 ELSE 0.0 END) as smoker_rate,
    AVG(age) as avg_age,
    AVG(bmi) as avg_bmi,
    
    -- Volume alerts
    CASE WHEN COUNT(*) < 10 THEN 'LOW_VOLUME' ELSE 'OK' END as volume_status
FROM ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY DATE(prediction_timestamp)
```

### **2. Performance Summary Monitoring**
**View**: `juan_dev.healthcare_data.ml_monitoring_summary`

```sql
-- Aggregated performance metrics over 7 days
SELECT 
    COUNT(*) as total_predictions,
    COUNT(DISTINCT DATE(prediction_timestamp)) as active_days,
    AVG(adjusted_prediction) as avg_risk_score,
    
    -- Risk distribution tracking
    COUNT(CASE WHEN adjusted_prediction > 80 THEN 1 END) as high_risk_count,
    COUNT(CASE WHEN adjusted_prediction BETWEEN 50 AND 80 THEN 1 END) as medium_risk_count,
    COUNT(CASE WHEN adjusted_prediction < 50 THEN 1 END) as low_risk_count
FROM ml_patient_predictions
WHERE prediction_timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
```

### **3. Alert System Monitoring**
**View**: `juan_dev.healthcare_data.ml_model_alerts`

```sql
-- Automated alert generation based on thresholds
SELECT 
    prediction_date,
    CASE 
        WHEN volume_status = 'LOW_VOLUME' THEN 'VOLUME_ALERT'
        WHEN avg_prediction > 85 THEN 'HIGH_RISK_ALERT'
        WHEN avg_prediction < 15 THEN 'LOW_RISK_ALERT'
        ELSE 'NORMAL'
    END as alert_type,
    
    CASE 
        WHEN volume_status = 'LOW_VOLUME' THEN 'MEDIUM'
        WHEN avg_prediction > 90 OR avg_prediction < 10 THEN 'HIGH'
        ELSE 'LOW'
    END as alert_severity
FROM ml_drift_monitor
WHERE volume_status != 'OK' OR avg_prediction > 85 OR avg_prediction < 15
```

### **4. Native Lakehouse Monitoring**
**Configuration**:
```python
MonitorInferenceLog(
    granularities=["1 day"],
    model_id_col="model_name",
    prediction_col="adjusted_prediction", 
    timestamp_col="prediction_timestamp",
    problem_type="PROBLEM_TYPE_REGRESSION"
)
```

## Monitoring Dimensions

### **Data Quality Monitoring**
- **Volume**: Daily prediction counts with low-volume alerts
- **Distribution**: Statistical analysis of prediction ranges
- **Completeness**: Validates table accessibility and data availability

### **Model Performance Monitoring** 
- **Prediction Drift**: Tracks changes in average risk scores over time
- **Feature Drift**: Monitors shifts in input demographics (age, BMI, smoker rates)
- **Temporal Patterns**: Daily aggregations to identify trends

### **Operational Monitoring**
- **Throughput**: Monitors prediction volume and frequency
- **Latency**: Tracks prediction timestamps
- **Availability**: Ensures monitoring tables are accessible

### **Business Logic Monitoring**
- **Risk Segmentation**: Categorizes predictions into low/medium/high risk buckets
- **Threshold Alerts**: Flags unusual risk score distributions
- **Health Checks**: Validates model is operating within expected parameters

## Alert Thresholds

- **Volume Alert**: < 10 predictions per day
- **High Risk Alert**: Average daily risk score > 85
- **Low Risk Alert**: Average daily risk score < 15
- **Critical Alert**: Risk scores > 90 or < 10

This comprehensive monitoring setup provides both statistical drift detection and business-relevant alerting for the insurance risk prediction model.
