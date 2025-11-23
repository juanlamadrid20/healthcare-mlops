"""
Custom Metrics for Healthcare ML Model Monitoring

This module provides custom metric calculations for healthcare-specific
monitoring requirements including fairness, bias detection, and business KPIs.

Author: Healthcare MLOps Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, count, avg, stddev, min as spark_min, max as spark_max
from pyspark.sql.functions import when, sum as spark_sum, expr, lit, current_timestamp


class FairnessMetricsCalculator:
    """
    Calculate fairness and bias metrics for healthcare predictions.
    
    Monitors demographic parity, regional equity, and protected attribute fairness
    to ensure the model doesn't discriminate across sensitive groups.
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize the fairness metrics calculator.
        
        Args:
            spark: Active SparkSession
        """
        self.spark = spark
        
    def calculate_demographic_parity(
        self, 
        predictions_df: DataFrame,
        sensitive_attribute: str,
        prediction_col: str = "adjusted_prediction",
        threshold: float = 75.0
    ) -> pd.DataFrame:
        """
        Calculate demographic parity metrics.
        
        Measures whether high-risk predictions are equally distributed across
        demographic groups. Significant disparities may indicate bias.
        
        Args:
            predictions_df: DataFrame with predictions and demographics
            sensitive_attribute: Column name for demographic attribute (e.g., 'patient_gender')
            prediction_col: Column containing model predictions
            threshold: Threshold for defining "high-risk" predictions
            
        Returns:
            DataFrame with parity metrics by group
        """
        parity_metrics = (
            predictions_df
            .groupBy(sensitive_attribute)
            .agg(
                count("*").alias("total_count"),
                avg(prediction_col).alias("avg_prediction"),
                stddev(prediction_col).alias("stddev_prediction"),
                spark_sum(when(col(prediction_col) > threshold, 1).otherwise(0)).alias("high_risk_count"),
                (spark_sum(when(col(prediction_col) > threshold, 1).otherwise(0)) / count("*") * 100).alias("high_risk_rate")
            )
        )
        
        return parity_metrics.toPandas()
    
    def calculate_regional_fairness(
        self,
        predictions_df: DataFrame,
        region_col: str = "patient_region",
        prediction_col: str = "adjusted_prediction"
    ) -> pd.DataFrame:
        """
        Calculate regional fairness metrics.
        
        Ensures predictions are consistent across geographic regions,
        preventing location-based discrimination.
        
        Args:
            predictions_df: DataFrame with predictions and region info
            region_col: Column name for region
            prediction_col: Column containing model predictions
            
        Returns:
            DataFrame with regional fairness metrics
        """
        regional_metrics = (
            predictions_df
            .groupBy(region_col)
            .agg(
                count("*").alias("prediction_count"),
                avg(prediction_col).alias("avg_risk_score"),
                stddev(prediction_col).alias("stddev_risk_score"),
                spark_min(prediction_col).alias("min_risk_score"),
                spark_max(prediction_col).alias("max_risk_score")
            )
        )
        
        return regional_metrics.toPandas()
    
    def calculate_fairness_disparity(
        self,
        predictions_df: DataFrame,
        protected_attributes: List[str],
        prediction_col: str = "adjusted_prediction"
    ) -> Dict[str, float]:
        """
        Calculate overall fairness disparity scores.
        
        Computes the coefficient of variation across protected groups
        to measure overall fairness. Lower values indicate better fairness.
        
        Args:
            predictions_df: DataFrame with predictions
            protected_attributes: List of protected attribute columns
            prediction_col: Column containing model predictions
            
        Returns:
            Dictionary of disparity scores by attribute
        """
        disparity_scores = {}
        
        for attribute in protected_attributes:
            # Calculate coefficient of variation of average predictions across groups
            group_stats = (
                predictions_df
                .groupBy(attribute)
                .agg(avg(prediction_col).alias("group_avg"))
                .toPandas()
            )
            
            if len(group_stats) > 0:
                cv = group_stats["group_avg"].std() / group_stats["group_avg"].mean()
                disparity_scores[f"{attribute}_disparity"] = float(cv)
            else:
                disparity_scores[f"{attribute}_disparity"] = 0.0
                
        return disparity_scores
    
    def generate_fairness_report(
        self,
        predictions_df: DataFrame,
        output_table: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive fairness report.
        
        Args:
            predictions_df: DataFrame with predictions
            output_table: Optional Unity Catalog table to save results
            
        Returns:
            DataFrame with complete fairness analysis
        """
        # Calculate metrics for all protected attributes
        gender_parity = self.calculate_demographic_parity(
            predictions_df, 
            "patient_gender"
        )
        gender_parity["protected_attribute"] = "gender"
        
        region_fairness = self.calculate_regional_fairness(predictions_df)
        region_fairness.rename(columns={
            "patient_region": "group_value",
            "prediction_count": "total_count"
        }, inplace=True)
        region_fairness["protected_attribute"] = "region"
        
        # Calculate disparity scores
        disparity_scores = self.calculate_fairness_disparity(
            predictions_df,
            ["patient_gender", "patient_region", "patient_age_category", "patient_smoking_status"]
        )
        
        # Combine into comprehensive report
        fairness_report = pd.DataFrame([{
            "metric_timestamp": pd.Timestamp.now(),
            "total_predictions": predictions_df.count(),
            **disparity_scores,
            "fairness_threshold_violation": any(v > 0.15 for v in disparity_scores.values())
        }])
        
        # Save to Unity Catalog if requested
        if output_table:
            report_spark_df = self.spark.createDataFrame(fairness_report)
            report_spark_df.write.mode("append").saveAsTable(output_table)
            
        return fairness_report


class BusinessMetricsCalculator:
    """
    Calculate healthcare-specific business metrics.
    
    Tracks operational KPIs, risk distribution, and clinical decision support metrics
    relevant to the healthcare insurance domain.
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize the business metrics calculator.
        
        Args:
            spark: Active SparkSession
        """
        self.spark = spark
        
    def calculate_risk_distribution(
        self,
        predictions_df: DataFrame,
        risk_category_col: str = "risk_category"
    ) -> pd.DataFrame:
        """
        Calculate risk category distribution.
        
        Args:
            predictions_df: DataFrame with risk categories
            risk_category_col: Column containing risk categories
            
        Returns:
            DataFrame with risk distribution metrics
        """
        risk_dist = (
            predictions_df
            .groupBy(risk_category_col)
            .agg(
                count("*").alias("count")
            )
            .withColumn("percentage", col("count") / lit(predictions_df.count()) * 100)
            .orderBy(risk_category_col)
        )
        
        return risk_dist.toPandas()
    
    def calculate_prediction_quality_metrics(
        self,
        predictions_df: DataFrame,
        prediction_col: str = "adjusted_prediction",
        confidence_lower_col: str = "prediction_lower_bound",
        confidence_upper_col: str = "prediction_upper_bound"
    ) -> Dict[str, float]:
        """
        Calculate prediction quality and confidence metrics.
        
        Args:
            predictions_df: DataFrame with predictions and confidence intervals
            prediction_col: Column with predictions
            confidence_lower_col: Lower confidence bound column
            confidence_upper_col: Upper confidence bound column
            
        Returns:
            Dictionary of quality metrics
        """
        stats = predictions_df.select(
            avg(prediction_col).alias("mean_prediction"),
            stddev(prediction_col).alias("std_prediction"),
            spark_min(prediction_col).alias("min_prediction"),
            spark_max(prediction_col).alias("max_prediction"),
            avg((col(confidence_upper_col) - col(confidence_lower_col))).alias("avg_confidence_width")
        ).collect()[0]
        
        return {
            "mean_prediction": float(stats["mean_prediction"]),
            "std_prediction": float(stats["std_prediction"]),
            "min_prediction": float(stats["min_prediction"]),
            "max_prediction": float(stats["max_prediction"]),
            "avg_confidence_width": float(stats["avg_confidence_width"])
        }
    
    def calculate_throughput_metrics(
        self,
        predictions_df: DataFrame,
        timestamp_col: str = "prediction_timestamp",
        window_days: int = 7
    ) -> Dict[str, float]:
        """
        Calculate operational throughput metrics.
        
        Args:
            predictions_df: DataFrame with predictions
            timestamp_col: Column with prediction timestamps
            window_days: Number of days to analyze
            
        Returns:
            Dictionary of throughput metrics
        """
        total_predictions = predictions_df.count()
        
        # Calculate daily averages
        daily_predictions = total_predictions / window_days
        
        # Calculate predictions by day of week
        dow_stats = (
            predictions_df
            .withColumn("day_of_week", expr(f"dayofweek({timestamp_col})"))
            .groupBy("day_of_week")
            .agg(count("*").alias("count"))
            .toPandas()
        )
        
        return {
            "total_predictions": float(total_predictions),
            "daily_avg_predictions": float(daily_predictions),
            "max_daily_predictions": float(dow_stats["count"].max()) if len(dow_stats) > 0 else 0.0,
            "min_daily_predictions": float(dow_stats["count"].min()) if len(dow_stats) > 0 else 0.0
        }
    
    def generate_business_report(
        self,
        predictions_df: DataFrame,
        output_table: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive business metrics report.
        
        Args:
            predictions_df: DataFrame with predictions
            output_table: Optional Unity Catalog table to save results
            
        Returns:
            DataFrame with complete business metrics
        """
        # Calculate all business metrics
        risk_dist = self.calculate_risk_distribution(predictions_df)
        quality_metrics = self.calculate_prediction_quality_metrics(predictions_df)
        throughput_metrics = self.calculate_throughput_metrics(predictions_df)
        
        # High-risk patient metrics
        high_risk_count = predictions_df.filter(col("high_risk_patient") == True).count()
        requires_review_count = predictions_df.filter(col("requires_review") == True).count()
        total_count = predictions_df.count()
        
        # Combine into report
        business_report = pd.DataFrame([{
            "metric_timestamp": pd.Timestamp.now(),
            "total_predictions": total_count,
            "high_risk_count": high_risk_count,
            "high_risk_percentage": (high_risk_count / total_count * 100) if total_count > 0 else 0,
            "requires_review_count": requires_review_count,
            "requires_review_percentage": (requires_review_count / total_count * 100) if total_count > 0 else 0,
            **quality_metrics,
            **throughput_metrics
        }])
        
        # Save to Unity Catalog if requested
        if output_table:
            report_spark_df = self.spark.createDataFrame(business_report)
            report_spark_df.write.mode("append").saveAsTable(output_table)
            
        return business_report


class DriftDetector:
    """
    Detect statistical drift in predictions and features.
    
    Uses Population Stability Index (PSI) and Kullback-Leibler divergence
    to detect distribution shifts over time.
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize the drift detector.
        
        Args:
            spark: Active SparkSession
        """
        self.spark = spark
        
    def calculate_psi(
        self,
        baseline_df: DataFrame,
        current_df: DataFrame,
        column: str,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI measures the shift in a distribution between two time periods.
        PSI < 0.1: No significant change
        PSI 0.1-0.25: Moderate change, investigate
        PSI > 0.25: Significant change, retraining recommended
        
        Args:
            baseline_df: Baseline (reference) DataFrame
            current_df: Current DataFrame to compare
            column: Column name to analyze
            bins: Number of bins for discretization
            
        Returns:
            PSI score
        """
        # Convert to pandas for PSI calculation
        baseline_data = baseline_df.select(column).toPandas()[column].dropna()
        current_data = current_df.select(column).toPandas()[column].dropna()
        
        if len(baseline_data) == 0 or len(current_data) == 0:
            return 0.0
            
        # Create bins based on baseline distribution
        bins_edges = np.percentile(baseline_data, np.linspace(0, 100, bins + 1))
        bins_edges[-1] += 0.001  # Ensure max value is included
        
        # Calculate distributions
        baseline_dist, _ = np.histogram(baseline_data, bins=bins_edges)
        current_dist, _ = np.histogram(current_data, bins=bins_edges)
        
        # Normalize to percentages
        baseline_pct = baseline_dist / len(baseline_data)
        current_pct = current_dist / len(current_data)
        
        # Avoid division by zero
        baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
        current_pct = np.where(current_pct == 0, 0.0001, current_pct)
        
        # Calculate PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        
        return float(psi)
    
    def calculate_kl_divergence(
        self,
        baseline_df: DataFrame,
        current_df: DataFrame,
        column: str,
        bins: int = 10
    ) -> float:
        """
        Calculate Kullback-Leibler divergence.
        
        Measures how one probability distribution diverges from a reference distribution.
        
        Args:
            baseline_df: Baseline (reference) DataFrame
            current_df: Current DataFrame to compare
            column: Column name to analyze
            bins: Number of bins for discretization
            
        Returns:
            KL divergence score
        """
        # Convert to pandas
        baseline_data = baseline_df.select(column).toPandas()[column].dropna()
        current_data = current_df.select(column).toPandas()[column].dropna()
        
        if len(baseline_data) == 0 or len(current_data) == 0:
            return 0.0
            
        # Create bins
        bins_edges = np.percentile(baseline_data, np.linspace(0, 100, bins + 1))
        bins_edges[-1] += 0.001
        
        # Calculate distributions
        baseline_dist, _ = np.histogram(baseline_data, bins=bins_edges)
        current_dist, _ = np.histogram(current_data, bins=bins_edges)
        
        # Normalize
        baseline_prob = baseline_dist / baseline_dist.sum()
        current_prob = current_dist / current_dist.sum()
        
        # Avoid log(0)
        baseline_prob = np.where(baseline_prob == 0, 1e-10, baseline_prob)
        current_prob = np.where(current_prob == 0, 1e-10, current_prob)
        
        # Calculate KL divergence
        kl_div = np.sum(current_prob * np.log(current_prob / baseline_prob))
        
        return float(kl_div)
    
    def detect_drift_multi_column(
        self,
        baseline_df: DataFrame,
        current_df: DataFrame,
        columns: List[str],
        output_table: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Detect drift across multiple columns.
        
        Args:
            baseline_df: Baseline DataFrame
            current_df: Current DataFrame
            columns: List of columns to analyze
            output_table: Optional Unity Catalog table to save results
            
        Returns:
            DataFrame with drift metrics for each column
        """
        drift_results = []
        
        for column in columns:
            try:
                psi_score = self.calculate_psi(baseline_df, current_df, column)
                kl_score = self.calculate_kl_divergence(baseline_df, current_df, column)
                
                # Determine drift severity
                if psi_score < 0.1:
                    severity = "no_drift"
                elif psi_score < 0.25:
                    severity = "moderate_drift"
                else:
                    severity = "significant_drift"
                    
                drift_results.append({
                    "column_name": column,
                    "psi_score": psi_score,
                    "kl_divergence": kl_score,
                    "drift_severity": severity,
                    "requires_action": psi_score > 0.25,
                    "metric_timestamp": pd.Timestamp.now()
                })
            except Exception as e:
                print(f"Error calculating drift for column {column}: {str(e)}")
                drift_results.append({
                    "column_name": column,
                    "psi_score": None,
                    "kl_divergence": None,
                    "drift_severity": "error",
                    "requires_action": False,
                    "metric_timestamp": pd.Timestamp.now()
                })
        
        drift_df = pd.DataFrame(drift_results)
        
        # Save to Unity Catalog if requested
        if output_table:
            drift_spark_df = self.spark.createDataFrame(drift_df)
            drift_spark_df.write.mode("append").saveAsTable(output_table)
            
        return drift_df


def calculate_all_custom_metrics(
    spark: SparkSession,
    predictions_table: str,
    baseline_table: str,
    output_schema: str,
    catalog: str = "juan_dev",
    schema: str = "healthcare_data"
) -> Dict[str, pd.DataFrame]:
    """
    Calculate all custom metrics in one orchestrated function.
    
    Args:
        spark: Active SparkSession
        predictions_table: Full name of predictions table
        baseline_table: Full name of baseline table (e.g., dim_patients)
        output_schema: Schema to save custom metric tables
        catalog: Unity Catalog catalog name
        schema: Unity Catalog schema name
        
    Returns:
        Dictionary containing all metric DataFrames
    """
    # Load data
    predictions_df = spark.table(predictions_table)
    baseline_df = spark.table(baseline_table)
    
    # Initialize calculators
    fairness_calc = FairnessMetricsCalculator(spark)
    business_calc = BusinessMetricsCalculator(spark)
    drift_detector = DriftDetector(spark)
    
    # Calculate all metrics
    print("Calculating fairness metrics...")
    fairness_report = fairness_calc.generate_fairness_report(
        predictions_df,
        output_table=f"{catalog}.{schema}.custom_fairness_metrics"
    )
    
    print("Calculating business metrics...")
    business_report = business_calc.generate_business_report(
        predictions_df,
        output_table=f"{catalog}.{schema}.custom_business_metrics"
    )
    
    print("Detecting drift...")
    
    # Strategy: Compare predictions table against baseline for common demographic features
    # This detects if the input population has shifted
    # Note: PSI/KL divergence only work on numerical columns
    
    # Get columns that exist in both dataframes
    predictions_cols = set(predictions_df.columns)
    baseline_cols = set(baseline_df.columns)
    
    # Common numerical columns for population drift detection
    # Categorical columns like patient_age_category and patient_smoking_status 
    # cannot be used with PSI (requires numerical data for histogram binning)
    common_drift_cols = ["bmi", "health_risk_score"]
    drift_columns = [c for c in common_drift_cols if c in predictions_cols and c in baseline_cols]
    
    print(f"Analyzing population drift on numerical columns: {drift_columns}")
    
    if len(drift_columns) > 0:
        drift_report = drift_detector.detect_drift_multi_column(
            baseline_df.filter(col("is_current_record") == True) if "is_current_record" in baseline_cols else baseline_df,
            predictions_df,
            drift_columns,
            output_table=f"{catalog}.{schema}.drift_analysis_summary"
        )
    else:
        print("Warning: No common numerical columns found for drift detection")
        drift_report = pd.DataFrame({
            "column_name": ["no_common_columns"],
            "psi_score": [0.0],
            "kl_divergence": [0.0],
            "drift_severity": ["no_drift"],
            "requires_action": [False],
            "metric_timestamp": [pd.Timestamp.now()]
        })
    
    print("Custom metrics calculation complete!")
    
    return {
        "fairness": fairness_report,
        "business": business_report,
        "drift": drift_report
    }

