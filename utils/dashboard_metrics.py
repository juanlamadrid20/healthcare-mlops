# Databricks notebook source
"""
Dashboard Metrics Generator for Healthcare Insurance MLOps

This script generates key business metrics for executive dashboards and monitoring systems.
Metrics are designed to be consumed by BI tools, alerting systems, and executive reports.

Usage:
  - Run directly in Databricks to generate current metrics
  - Schedule as a job for regular metric updates
  - Export metrics to Delta tables for historical tracking
  - Integrate with alerting systems for anomaly detection

Output:
  - JSON formatted metrics for API consumption
  - Delta table with historical metrics
  - Alert triggers for anomaly detection
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import *

# COMMAND ----------

# Configuration
class DashboardMetricsConfig:
    """Configuration for dashboard metrics generation"""

    def __init__(self):
        # Get parameters from widgets (with defaults)
        try:
            self.catalog = dbutils.widgets.get("catalog")
            self.ml_schema = dbutils.widgets.get("ml_schema")
            self.model_name = dbutils.widgets.get("model_name")
            self.lookback_days = self.int(dbutils.widgets.get("lookback_days"))
            self.output_format = dbutils.widgets.get("output_format")
        except:
            # Default values if widgets not set
            self.catalog = "juan_dev"
            self.ml_schema = "healthcare_data"
            self.model_name = "insurance_model"
            self.lookback_days = 7
            self.output_format = "json"  # Options: json, delta, both

        # Derived attributes
        self.predictions_table = f"{self.catalog}.{self.ml_schema}.ml_patient_predictions"
        self.metrics_table = f"{self.catalog}.{self.ml_schema}.ml_dashboard_metrics"
        self.full_model_name = f"{self.catalog}.{self.ml_schema}.{self.model_name}"

        # Alert thresholds
        self.alert_thresholds = {
            "high_risk_pct_min": 5.0,
            "high_risk_pct_max": 30.0,
            "avg_risk_min": 20.0,
            "avg_risk_max": 70.0,
            "prediction_volume_min": 100,
            "null_prediction_pct_max": 1.0,
            "regional_disparity_max": 25.0
        }

# Initialize configuration
config = DashboardMetricsConfig()

# Setup widgets
try:
    dbutils.widgets.text("catalog", config.catalog, "Unity Catalog")
    dbutils.widgets.text("ml_schema", config.ml_schema, "ML Schema")
    dbutils.widgets.text("model_name", config.model_name, "Model Name")
    dbutils.widgets.text("lookback_days", str(config.lookback_days), "Lookback Days")
    dbutils.widgets.dropdown("output_format", config.output_format, ["json", "delta", "both"], "Output Format")
except:
    pass

print(f"Dashboard Metrics Configuration:")
print(f"  Predictions Table: {config.predictions_table}")
print(f"  Lookback Period: {config.lookback_days} days")
print(f"  Output Format: {config.output_format}")

# COMMAND ----------

class HealthcareMetricsGenerator:
    """Generate comprehensive dashboard metrics for healthcare model"""

    def __init__(self, config: DashboardMetricsConfig):
        self.config = config
        self.spark = SparkSession.builder.getOrCreate()
        mlflow.set_registry_uri("databricks-uc")
        self.mlflow_client = MlflowClient()
        self.metrics = {}
        self.alerts = []
        # Import builtins to avoid conflicts with PySpark functions
        import builtins as _builtins
        self.round = _builtins.round
        self.max = _builtins.max
        self.min = _builtins.min
        self.abs = _builtins.abs
        self.sum = _builtins.sum
        self.int = _builtins.int

    def generate_all_metrics(self) -> Dict[str, Any]:
        """Generate all dashboard metrics"""
        print("=" * 80)
        print("GENERATING DASHBOARD METRICS")
        print("=" * 80)

        self.metrics["metadata"] = self._get_metadata()
        self.metrics["volume_metrics"] = self._get_volume_metrics()
        self.metrics["risk_metrics"] = self._get_risk_metrics()
        self.metrics["demographic_metrics"] = self._get_demographic_metrics()
        self.metrics["regional_metrics"] = self._get_regional_metrics()
        self.metrics["quality_metrics"] = self._get_quality_metrics()
        self.metrics["model_metrics"] = self._get_model_metrics()
        self.metrics["trend_metrics"] = self._get_trend_metrics()
        self.metrics["business_metrics"] = self._get_business_metrics()
        self.metrics["alerts"] = self.alerts

        print("\n✅ All metrics generated successfully")
        return self.metrics

    def _get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the metrics generation"""
        return {
            "generated_at": datetime.now().isoformat(),
            "lookback_days": self.config.lookback_days,
            "model_name": self.config.full_model_name,
            "predictions_table": self.config.predictions_table
        }

    def _get_volume_metrics(self) -> Dict[str, Any]:
        """Get prediction volume metrics"""
        print("\n📊 Calculating volume metrics...")

        predictions_df = self.spark.table(self.config.predictions_table)
        cutoff_date = (datetime.now() - timedelta(days=self.config.lookback_days)).strftime('%Y-%m-%d')
        recent_predictions = predictions_df.filter(col("prediction_timestamp") >= cutoff_date)

        volume_stats = recent_predictions.agg(
            count("*").alias("total_predictions"),
            countDistinct("customer_id").alias("unique_patients"),
            F.min("prediction_timestamp").alias("earliest_prediction"),
            F.max("prediction_timestamp").alias("latest_prediction")
        ).collect()[0]

        metrics = {
            "total_predictions": self.int(volume_stats.total_predictions),
            "unique_patients": self.int(volume_stats.unique_patients),
            "earliest_prediction": str(volume_stats.earliest_prediction),
            "latest_prediction": str(volume_stats.latest_prediction),
            "avg_predictions_per_day": self.round(volume_stats.total_predictions / self.config.lookback_days, 2)
        }

        # Alert: Low prediction volume
        if metrics["total_predictions"] < self.config.alert_thresholds["prediction_volume_min"]:
            self.alerts.append({
                "type": "LOW_VOLUME",
                "severity": "WARNING",
                "message": f"Low prediction volume: {metrics['total_predictions']} < {self.config.alert_thresholds['prediction_volume_min']}"
            })

        return metrics

    def _get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk score metrics"""
        print("📈 Calculating risk metrics...")

        predictions_df = self.spark.table(self.config.predictions_table)
        cutoff_date = (datetime.now() - timedelta(days=self.config.lookback_days)).strftime('%Y-%m-%d')
        recent_predictions = predictions_df.filter(col("prediction_timestamp") >= cutoff_date)

        risk_stats = recent_predictions.agg(
            avg("adjusted_prediction").alias("avg_risk"),
            stddev("adjusted_prediction").alias("risk_std"),
            min("adjusted_prediction").alias("min_risk"),
            max("adjusted_prediction").alias("max_risk"),
            expr("percentile(adjusted_prediction, 0.5)").alias("median_risk"),
            expr("percentile(adjusted_prediction, 0.95)").alias("p95_risk"),
            (sum(when(col("high_risk_patient") == True, 1).otherwise(0)) / count("*") * 100).alias("high_risk_pct")
        ).collect()[0]

        # Risk category distribution
        risk_dist = recent_predictions.groupBy("risk_category").agg(
            count("*").alias("count")
        ).collect()

        total_preds = self.sum([row['count'] for row in risk_dist])
        risk_distribution = {
            row.risk_category: {
                "count": self.int(row['count']),
                "percentage": self.round((row['count'] / total_preds) * 100, 2)
            } for row in risk_dist
        }

        metrics = {
            "avg_risk_score": self.round(risk_stats.avg_risk, 2),
            "risk_std_dev": self.round(risk_stats.risk_std, 2),
            "min_risk": self.round(risk_stats.min_risk, 2),
            "max_risk": self.round(risk_stats.max_risk, 2),
            "median_risk": self.round(risk_stats.median_risk, 2),
            "p95_risk": self.round(risk_stats.p95_risk, 2),
            "high_risk_percentage": self.round(risk_stats.high_risk_pct, 2),
            "risk_distribution": risk_distribution
        }

        # Alerts: Risk score anomalies
        if metrics["avg_risk_score"] < self.config.alert_thresholds["avg_risk_min"]:
            self.alerts.append({
                "type": "LOW_AVG_RISK",
                "severity": "WARNING",
                "message": f"Average risk unusually low: {metrics['avg_risk_score']}"
            })
        elif metrics["avg_risk_score"] > self.config.alert_thresholds["avg_risk_max"]:
            self.alerts.append({
                "type": "HIGH_AVG_RISK",
                "severity": "WARNING",
                "message": f"Average risk unusually high: {metrics['avg_risk_score']}"
            })

        if metrics["high_risk_percentage"] < self.config.alert_thresholds["high_risk_pct_min"]:
            self.alerts.append({
                "type": "LOW_HIGH_RISK_PCT",
                "severity": "INFO",
                "message": f"Low high-risk percentage: {metrics['high_risk_percentage']}%"
            })
        elif metrics["high_risk_percentage"] > self.config.alert_thresholds["high_risk_pct_max"]:
            self.alerts.append({
                "type": "HIGH_HIGH_RISK_PCT",
                "severity": "WARNING",
                "message": f"High high-risk percentage: {metrics['high_risk_percentage']}%"
            })

        return metrics

    def _get_demographic_metrics(self) -> Dict[str, Any]:
        """Get demographic breakdown metrics"""
        print("👥 Calculating demographic metrics...")

        predictions_df = self.spark.table(self.config.predictions_table)
        cutoff_date = (datetime.now() - timedelta(days=self.config.lookback_days)).strftime('%Y-%m-%d')
        recent_predictions = predictions_df.filter(col("prediction_timestamp") >= cutoff_date)

        # Smoking status breakdown
        smoking_metrics = recent_predictions.groupBy("patient_smoking_status").agg(
            count("*").alias("count"),
            avg("adjusted_prediction").alias("avg_risk"),
            (sum(when(col("high_risk_patient") == True, 1).otherwise(0)) / count("*") * 100).alias("high_risk_pct")
        ).collect()

        # Age breakdown
        age_metrics = recent_predictions.groupBy("patient_age_category").agg(
            count("*").alias("count"),
            avg("adjusted_prediction").alias("avg_risk")
        ).collect()

        # Gender breakdown
        gender_metrics = recent_predictions.groupBy("sex").agg(
            count("*").alias("count"),
            avg("adjusted_prediction").alias("avg_risk"),
            (sum(when(col("high_risk_patient") == True, 1).otherwise(0)) / count("*") * 100).alias("high_risk_pct")
        ).collect()

        metrics = {
            "smoking_status": {
                row.patient_smoking_status: {
                    "count": self.int(row['count']),
                    "avg_risk": self.round(row.avg_risk, 2),
                    "high_risk_pct": self.round(row.high_risk_pct, 2)
                } for row in smoking_metrics
            },
            "age_category": {
                row.patient_age_category: {
                    "count": self.int(row['count']),
                    "avg_risk": self.round(row.avg_risk, 2)
                } for row in age_metrics
            },
            "gender": {
                row.sex: {
                    "count": self.int(row['count']),
                    "avg_risk": self.round(row.avg_risk, 2),
                    "high_risk_pct": self.round(row.high_risk_pct, 2)
                } for row in gender_metrics
            }
        }

        return metrics

    def _get_regional_metrics(self) -> Dict[str, Any]:
        """Get regional equity metrics"""
        print("🌎 Calculating regional metrics...")

        predictions_df = self.spark.table(self.config.predictions_table)
        cutoff_date = (datetime.now() - timedelta(days=self.config.lookback_days)).strftime('%Y-%m-%d')
        recent_predictions = predictions_df.filter(col("prediction_timestamp") >= cutoff_date)

        regional_stats = recent_predictions.groupBy("region").agg(
            count("*").alias("count"),
            avg("adjusted_prediction").alias("avg_risk"),
            stddev("adjusted_prediction").alias("risk_std"),
            (sum(when(col("high_risk_patient") == True, 1).otherwise(0)) / count("*") * 100).alias("high_risk_pct")
        ).collect()

        regional_metrics = {
            row.region: {
                "count": self.int(row['count']),
                "avg_risk": self.round(row.avg_risk, 2),
                "risk_std": self.round(row.risk_std, 2) if row.risk_std else 0,
                "high_risk_pct": self.round(row.high_risk_pct, 2)
            } for row in regional_stats
        }

        # Calculate regional disparity using Python's built-in max/min
        regional_risks = [row['avg_risk'] for row in regional_metrics.values()]
        if len(regional_risks) > 1:
            disparity = ((self.max(regional_risks) - self.min(regional_risks)) / self.min(regional_risks)) * 100

            if disparity > self.config.alert_thresholds["regional_disparity_max"]:
                self.alerts.append({
                    "type": "HIGH_REGIONAL_DISPARITY",
                    "severity": "WARNING",
                    "message": f"High regional disparity: {disparity:.1f}%"
                })
        else:
            disparity = 0

        return {
            "by_region": regional_metrics,
            "regional_disparity_pct": self.round(disparity, 2)
        }

    def _get_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality metrics"""
        print("✅ Calculating quality metrics...")

        predictions_df = self.spark.table(self.config.predictions_table)
        cutoff_date = (datetime.now() - timedelta(days=self.config.lookback_days)).strftime('%Y-%m-%d')
        recent_predictions = predictions_df.filter(col("prediction_timestamp") >= cutoff_date)

        total_count = recent_predictions.count()

        quality_stats = recent_predictions.agg(
            (sum(when(col("adjusted_prediction").isNull(), 1).otherwise(0)) / count("*") * 100).alias("null_pct"),
            (sum(when(col("adjusted_prediction") < 0, 1).otherwise(0)) / count("*") * 100).alias("negative_pct"),
            (sum(when(col("adjusted_prediction") > 100, 1).otherwise(0)) / count("*") * 100).alias("outlier_pct"),
            avg(col("prediction_upper_bound") - col("prediction_lower_bound")).alias("avg_ci_width")
        ).collect()[0]

        metrics = {
            "null_prediction_pct": self.round(quality_stats.null_pct, 2),
            "negative_prediction_pct": self.round(quality_stats.negative_pct, 2),
            "outlier_prediction_pct": self.round(quality_stats.outlier_pct, 2),
            "avg_confidence_interval_width": self.round(quality_stats.avg_ci_width, 2),
            "data_completeness_pct": self.round(100 - quality_stats.null_pct, 2)
        }

        # Alerts: Data quality issues
        if metrics["null_prediction_pct"] > self.config.alert_thresholds["null_prediction_pct_max"]:
            self.alerts.append({
                "type": "HIGH_NULL_RATE",
                "severity": "ERROR",
                "message": f"High null prediction rate: {metrics['null_prediction_pct']}%"
            })

        return metrics

    def _get_model_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        print("🤖 Calculating model metrics...")

        try:
            champion_info = self.mlflow_client.get_model_version_by_alias(
                self.config.full_model_name, "champion"
            )

            run_data = self.mlflow_client.get_run(champion_info.run_id)
            mlflow_metrics = run_data.data.metrics

            metrics = {
                "champion_version": self.int(champion_info.version),
                "champion_status": champion_info.status,
                "r2_score": self.round(mlflow_metrics.get("r2_score", 0), 4),
                "mae": self.round(mlflow_metrics.get("mean_absolute_error", 0), 4),
                "rmse": self.round(mlflow_metrics.get("root_mean_squared_error", 0), 4),
                "high_risk_accuracy": self.round(mlflow_metrics.get("high_risk_accuracy", 0), 4)
            }
        except Exception as e:
            print(f"  ⚠️  Could not retrieve model metrics: {e}")
            metrics = {
                "champion_version": None,
                "error": str(e)
            }

        return metrics

    def _get_trend_metrics(self) -> Dict[str, Any]:
        """Get trend metrics comparing periods"""
        print("📊 Calculating trend metrics...")

        predictions_df = self.spark.table(self.config.predictions_table)

        # Current period
        current_cutoff = (datetime.now() - timedelta(days=self.config.lookback_days)).strftime('%Y-%m-%d')
        current_predictions = predictions_df.filter(col("prediction_timestamp") >= current_cutoff)

        # Previous period
        previous_cutoff = (datetime.now() - timedelta(days=self.config.lookback_days * 2)).strftime('%Y-%m-%d')
        previous_predictions = predictions_df.filter(
            (col("prediction_timestamp") >= previous_cutoff) &
            (col("prediction_timestamp") < current_cutoff)
        )

        current_stats = current_predictions.agg(
            count("*").alias("count"),
            avg("adjusted_prediction").alias("avg_risk"),
            (sum(when(col("high_risk_patient") == True, 1).otherwise(0)) / count("*") * 100).alias("high_risk_pct")
        ).collect()[0]

        previous_stats = previous_predictions.agg(
            count("*").alias("count"),
            avg("adjusted_prediction").alias("avg_risk"),
            (sum(when(col("high_risk_patient") == True, 1).otherwise(0)) / count("*") * 100).alias("high_risk_pct")
        ).collect()[0]

        if previous_stats['count'] > 0:
            volume_change_pct = ((current_stats['count'] - previous_stats['count']) / previous_stats['count']) * 100
            risk_change_pct = ((current_stats.avg_risk - previous_stats.avg_risk) / previous_stats.avg_risk) * 100
        else:
            volume_change_pct = 0
            risk_change_pct = 0

        metrics = {
            "current_period": {
                "volume": self.int(current_stats['count']),
                "avg_risk": self.round(current_stats.avg_risk, 2),
                "high_risk_pct": self.round(current_stats.high_risk_pct, 2)
            },
            "previous_period": {
                "volume": self.int(previous_stats['count']),
                "avg_risk": self.round(previous_stats.avg_risk, 2) if previous_stats['count'] > 0 else 0,
                "high_risk_pct": self.round(previous_stats.high_risk_pct, 2) if previous_stats['count'] > 0 else 0
            },
            "changes": {
                "volume_change_pct": self.round(volume_change_pct, 2),
                "risk_change_pct": self.round(risk_change_pct, 2)
            }
        }

        return metrics

    def _get_business_metrics(self) -> Dict[str, Any]:
        """Get business impact metrics"""
        print("💰 Calculating business metrics...")

        predictions_df = self.spark.table(self.config.predictions_table)
        cutoff_date = (datetime.now() - timedelta(days=self.config.lookback_days)).strftime('%Y-%m-%d')
        recent_predictions = predictions_df.filter(col("prediction_timestamp") >= cutoff_date)

        high_risk_count = recent_predictions.filter(col("high_risk_patient") == True).count()

        # Business assumptions (should be configurable)
        avg_intervention_cost = 500
        avg_prevented_claim = 5000
        intervention_success_rate = 0.30

        intervention_costs = high_risk_count * avg_intervention_cost
        successful_interventions = high_risk_count * intervention_success_rate
        prevented_claims_value = successful_interventions * avg_prevented_claim
        net_benefit = prevented_claims_value - intervention_costs

        metrics = {
            "high_risk_patients_identified": self.int(high_risk_count),
            "estimated_intervention_costs": self.round(intervention_costs, 2),
            "estimated_prevented_claims_value": self.round(prevented_claims_value, 2),
            "estimated_net_benefit": self.round(net_benefit, 2),
            "estimated_roi_pct": self.round((net_benefit / intervention_costs * 100), 2) if intervention_costs > 0 else 0
        }

        return metrics

    def export_metrics(self, output_format: str = None) -> None:
        """Export metrics to specified format"""
        if output_format is None:
            output_format = self.config.output_format

        if output_format in ["json", "both"]:
            self._export_json()

        if output_format in ["delta", "both"]:
            self._export_delta()

    def _export_json(self) -> None:
        """Export metrics as JSON"""
        json_output = json.dumps(self.metrics, indent=2)
        print("\n" + "=" * 80)
        print("METRICS JSON OUTPUT")
        print("=" * 80)
        print(json_output)

    def _export_delta(self) -> None:
        """Export metrics to Delta table for historical tracking"""
        print(f"\n📝 Exporting metrics to Delta table: {self.config.metrics_table}")

        # Flatten metrics for Delta table
        flat_metrics = {
            "metric_timestamp": datetime.now(),
            "lookback_days": self.config.lookback_days,
            "total_predictions": self.metrics["volume_metrics"]["total_predictions"],
            "unique_patients": self.metrics["volume_metrics"]["unique_patients"],
            "avg_risk_score": self.metrics["risk_metrics"]["avg_risk_score"],
            "high_risk_percentage": self.metrics["risk_metrics"]["high_risk_percentage"],
            "regional_disparity_pct": self.metrics["regional_metrics"]["regional_disparity_pct"],
            "data_completeness_pct": self.metrics["quality_metrics"]["data_completeness_pct"],
            "estimated_roi_pct": self.metrics["business_metrics"]["estimated_roi_pct"],
            "alert_count": len(self.alerts),
            "metrics_json": json.dumps(self.metrics)
        }

        # Create DataFrame
        metrics_df = self.spark.createDataFrame([flat_metrics])

        # Write to Delta table (append mode)
        metrics_df.write.format("delta").mode("append").saveAsTable(self.config.metrics_table)

        print(f"✅ Metrics exported to {self.config.metrics_table}")

# COMMAND ----------

# Generate and export metrics
generator = HealthcareMetricsGenerator(config)
metrics = generator.generate_all_metrics()
generator.export_metrics()

# COMMAND ----------

# Display summary
print("\n" + "=" * 80)
print("DASHBOARD METRICS SUMMARY")
print("=" * 80)
print(f"Total Predictions: {metrics['volume_metrics']['total_predictions']:,}")
print(f"Average Risk Score: {metrics['risk_metrics']['avg_risk_score']}")
print(f"High-Risk Percentage: {metrics['risk_metrics']['high_risk_percentage']}%")
print(f"Regional Disparity: {metrics['regional_metrics']['regional_disparity_pct']}%")
print(f"Data Completeness: {metrics['quality_metrics']['data_completeness_pct']}%")
print(f"Estimated ROI: {metrics['business_metrics']['estimated_roi_pct']}%")
print(f"Active Alerts: {len(metrics['alerts'])}")

if metrics['alerts']:
    print("\n⚠️  ACTIVE ALERTS:")
    for alert in metrics['alerts']:
        print(f"  [{alert['severity']}] {alert['type']}: {alert['message']}")

print("=" * 80)

# COMMAND ----------

# Return metrics as JSON for notebook exit
dbutils.notebook.exit(json.dumps({
    "status": "SUCCESS",
    "metrics_generated": True,
    "alert_count": len(metrics['alerts']),
    "high_risk_patients": metrics['business_metrics']['high_risk_patients_identified']
}))
