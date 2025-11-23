"""
Databricks Lakehouse Monitoring for Healthcare ML Model

This module provides comprehensive monitoring infrastructure using native
Databricks Lakehouse Monitoring APIs for healthcare insurance risk prediction models.

Features:
- Inference monitoring for model predictions
- Feature store quality monitoring
- Baseline data quality monitoring
- Custom healthcare-specific metrics
- Automated refresh scheduling
- Alert configuration

Author: Healthcare MLOps Team
Version: 1.0.0
"""

import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
    MonitorTimeSeries,
    MonitorSnapshot,
    MonitorCronSchedule,
    MonitorNotifications,
    MonitorDestination
)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


class HealthcareMonitorManager:
    """
    Manage Databricks Lakehouse Monitoring for healthcare ML models.
    
    This class handles the creation and configuration of monitors for:
    - Inference logs (predictions table)
    - Feature store (feature engineering output)
    - Baseline data (training data source)
    """
    
    def __init__(
        self,
        catalog: str = "juan_dev",
        schema: str = "healthcare_data",
        user_email: Optional[str] = None,
        workspace_client: Optional[WorkspaceClient] = None
    ):
        """
        Initialize the Healthcare Monitor Manager.
        
        Args:
            catalog: Unity Catalog catalog name
            schema: Unity Catalog schema name
            user_email: Email for notifications (defaults to current user)
            workspace_client: Optional WorkspaceClient instance
        """
        self.catalog = catalog
        self.schema = schema
        self.workspace_client = workspace_client or WorkspaceClient()
        
        # Get user email if not provided
        if user_email is None:
            try:
                self.user_email = self.workspace_client.current_user.me().user_name
            except:
                self.user_email = "admin@databricks.com"
        else:
            self.user_email = user_email
            
        # Define table names
        self.inference_table = f"{catalog}.{schema}.ml_patient_predictions"
        self.feature_table = f"{catalog}.{schema}.ml_insurance_features"
        self.baseline_table = f"{catalog}.{schema}.dim_patients"  # Use dim_patients (has dimension_last_updated)
        
        # Define output schema for monitoring tables
        self.output_schema = f"{catalog}.{schema}"
        
        print(f"Initialized HealthcareMonitorManager")
        print(f"  Catalog: {catalog}")
        print(f"  Schema: {schema}")
        print(f"  User: {self.user_email}")
        
    def create_inference_monitor(
        self,
        enable_baseline_comparison: bool = True,
        schedule_cron: Optional[str] = "0 0 6 * * ?",  # Daily at 6 AM
        notification_emails: Optional[List[str]] = None
    ) -> Dict:
        """
        Create an InferenceLog monitor for the predictions table.
        
        This monitor tracks:
        - Model prediction distributions over time
        - Prediction drift detection
        - Model performance metrics (when ground truth available)
        - Input feature drift
        
        Args:
            enable_baseline_comparison: Whether to enable baseline comparison
            schedule_cron: Cron expression for scheduled refreshes
            notification_emails: List of emails to notify on failures
            
        Returns:
            Dictionary with monitor creation details
        """
        print(f"\nCreating InferenceLog monitor for {self.inference_table}...")
        
        # Set up assets directory
        assets_dir = f"/Workspace/Users/{self.user_email}/databricks_lakehouse_monitoring/{self.inference_table}"
        
        # Configure notifications
        notifications = None
        if notification_emails:
            notifications = MonitorNotifications(
                on_failure=MonitorDestination(email_addresses=notification_emails)
            )
        elif self.user_email:
            notifications = MonitorNotifications(
                on_failure=MonitorDestination(email_addresses=[self.user_email])
            )
        
        # Configure schedule
        schedule = None
        if schedule_cron:
            schedule = MonitorCronSchedule(
                quartz_cron_expression=schedule_cron,
                timezone_id="UTC"
            )
        
        try:
            # Create the inference monitor
            monitor_info = self.workspace_client.quality_monitors.create(
                table_name=self.inference_table,
                assets_dir=assets_dir,
                output_schema_name=self.output_schema,
                inference_log=MonitorInferenceLog(
                    problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
                    prediction_col="adjusted_prediction",
                    timestamp_col="prediction_timestamp",
                    model_id_col="model_name",
                    granularities=["1 day", "1 week"],
                    # label_col can be added when ground truth becomes available
                    # label_col="actual_health_risk_score"
                ),
                schedule=schedule,
                notifications=notifications
            )
            
            print(f"✓ InferenceLog monitor created successfully!")
            print(f"  Assets directory: {assets_dir}")
            print(f"  Output schema: {self.output_schema}")
            print(f"  Schedule: {schedule_cron if schedule_cron else 'Manual only'}")
            
            return {
                "status": "created",
                "table_name": self.inference_table,
                "monitor_type": "InferenceLog",
                "assets_dir": assets_dir,
                "schedule": schedule_cron
            }
            
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"⚠ Monitor already exists for {self.inference_table}")
                return {
                    "status": "exists",
                    "table_name": self.inference_table,
                    "message": "Monitor already exists"
                }
            else:
                print(f"✗ Error creating monitor: {str(e)}")
                raise e
    
    def create_feature_monitor(
        self,
        schedule_cron: Optional[str] = "0 0 6 * * ?",  # Daily at 6 AM
        notification_emails: Optional[List[str]] = None
    ) -> Dict:
        """
        Create a Snapshot monitor for the feature table.
        
        This monitor tracks:
        - Feature distribution changes over time
        - Data quality of engineered features
        - Feature store health
        
        Args:
            schedule_cron: Cron expression for scheduled refreshes
            notification_emails: List of emails to notify on failures
            
        Returns:
            Dictionary with monitor creation details
        """
        print(f"\nCreating Snapshot monitor for {self.feature_table}...")
        
        # Set up assets directory
        assets_dir = f"/Workspace/Users/{self.user_email}/databricks_lakehouse_monitoring/{self.feature_table}"
        
        # Configure notifications
        notifications = None
        if notification_emails:
            notifications = MonitorNotifications(
                on_failure=MonitorDestination(email_addresses=notification_emails)
            )
        elif self.user_email:
            notifications = MonitorNotifications(
                on_failure=MonitorDestination(email_addresses=[self.user_email])
            )
        
        # Configure schedule
        schedule = None
        if schedule_cron:
            schedule = MonitorCronSchedule(
                quartz_cron_expression=schedule_cron,
                timezone_id="UTC"
            )
        
        try:
            # Create the snapshot monitor
            monitor_info = self.workspace_client.quality_monitors.create(
                table_name=self.feature_table,
                assets_dir=assets_dir,
                output_schema_name=self.output_schema,
                snapshot=MonitorSnapshot(),
                schedule=schedule,
                notifications=notifications
            )
            
            print(f"✓ Snapshot monitor created successfully!")
            print(f"  Assets directory: {assets_dir}")
            print(f"  Output schema: {self.output_schema}")
            
            return {
                "status": "created",
                "table_name": self.feature_table,
                "monitor_type": "Snapshot",
                "assets_dir": assets_dir,
                "schedule": schedule_cron
            }
            
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"⚠ Monitor already exists for {self.feature_table}")
                return {
                    "status": "exists",
                    "table_name": self.feature_table,
                    "message": "Monitor already exists"
                }
            else:
                print(f"✗ Error creating monitor: {str(e)}")
                raise e
    
    def create_baseline_monitor(
        self,
        schedule_cron: Optional[str] = "0 0 6 * * ?",  # Daily at 6 AM
        notification_emails: Optional[List[str]] = None
    ) -> Dict:
        """
        Create a TimeSeries monitor for the baseline training data.
        
        This monitor tracks:
        - Upstream data quality changes
        - Training data distribution over time
        - Data freshness and completeness
        
        Args:
            schedule_cron: Cron expression for scheduled refreshes
            notification_emails: List of emails to notify on failures
            
        Returns:
            Dictionary with monitor creation details
        """
        print(f"\nCreating TimeSeries monitor for {self.baseline_table}...")
        
        # Set up assets directory
        assets_dir = f"/Workspace/Users/{self.user_email}/databricks_lakehouse_monitoring/{self.baseline_table}"
        
        # Configure notifications
        notifications = None
        if notification_emails:
            notifications = MonitorNotifications(
                on_failure=MonitorDestination(email_addresses=notification_emails)
            )
        elif self.user_email:
            notifications = MonitorNotifications(
                on_failure=MonitorDestination(email_addresses=[self.user_email])
            )
        
        # Configure schedule
        schedule = None
        if schedule_cron:
            schedule = MonitorCronSchedule(
                quartz_cron_expression=schedule_cron,
                timezone_id="UTC"
            )
        
        try:
            # Create the time series monitor
            monitor_info = self.workspace_client.quality_monitors.create(
                table_name=self.baseline_table,
                assets_dir=assets_dir,
                output_schema_name=self.output_schema,
                time_series=MonitorTimeSeries(
                    timestamp_col="dimension_last_updated",
                    granularities=["1 day", "1 week"]
                ),
                schedule=schedule,
                notifications=notifications
            )
            
            print(f"✓ TimeSeries monitor created successfully!")
            print(f"  Assets directory: {assets_dir}")
            print(f"  Output schema: {self.output_schema}")
            
            return {
                "status": "created",
                "table_name": self.baseline_table,
                "monitor_type": "TimeSeries",
                "assets_dir": assets_dir,
                "schedule": schedule_cron
            }
            
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"⚠ Monitor already exists for {self.baseline_table}")
                return {
                    "status": "exists",
                    "table_name": self.baseline_table,
                    "message": "Monitor already exists"
                }
            else:
                print(f"✗ Error creating monitor: {str(e)}")
                raise e
    
    def create_all_monitors(
        self,
        schedule_cron: str = "0 0 6 * * ?",
        notification_emails: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Create all monitors in one operation.
        
        Args:
            schedule_cron: Cron expression for scheduled refreshes
            notification_emails: List of emails to notify on failures
            
        Returns:
            Dictionary with results for each monitor
        """
        print("=" * 80)
        print("Creating All Healthcare Monitoring Infrastructure")
        print("=" * 80)
        
        results = {}
        
        # Create inference monitor
        try:
            results["inference"] = self.create_inference_monitor(
                schedule_cron=schedule_cron,
                notification_emails=notification_emails
            )
        except Exception as e:
            results["inference"] = {"status": "error", "message": str(e)}
        
        # Create feature monitor
        try:
            results["feature"] = self.create_feature_monitor(
                schedule_cron=schedule_cron,
                notification_emails=notification_emails
            )
        except Exception as e:
            results["feature"] = {"status": "error", "message": str(e)}
        
        # Create baseline monitor
        try:
            results["baseline"] = self.create_baseline_monitor(
                schedule_cron=schedule_cron,
                notification_emails=notification_emails
            )
        except Exception as e:
            results["baseline"] = {"status": "error", "message": str(e)}
        
        print("\n" + "=" * 80)
        print("Monitor Creation Summary")
        print("=" * 80)
        for name, result in results.items():
            status_symbol = "✓" if result["status"] in ["created", "exists"] else "✗"
            print(f"{status_symbol} {name.capitalize()}: {result['status']}")
        
        return results
    
    def get_monitor_info(self, table_name: str) -> Dict:
        """
        Get information about an existing monitor.
        
        Args:
            table_name: Full table name
            
        Returns:
            Monitor configuration details
        """
        try:
            monitor = self.workspace_client.quality_monitors.get(table_name)
            return {
                "table_name": table_name,
                "status": monitor.status,
                "monitor_version": monitor.monitor_version,
                "drift_metrics_table_name": monitor.drift_metrics_table_name,
                "profile_metrics_table_name": monitor.profile_metrics_table_name
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_monitor(self, table_name: str) -> Dict:
        """
        Delete a monitor.
        
        Note: This does not delete the metric tables or dashboards.
        
        Args:
            table_name: Full table name
            
        Returns:
            Deletion status
        """
        try:
            self.workspace_client.quality_monitors.delete(table_name=table_name)
            print(f"✓ Monitor deleted for {table_name}")
            return {"status": "deleted", "table_name": table_name}
        except Exception as e:
            print(f"✗ Error deleting monitor: {str(e)}")
            return {"status": "error", "message": str(e)}


class MonitorRefreshManager:
    """
    Manage monitor refresh operations and scheduling.
    
    This class handles:
    - Manual refresh triggering
    - Refresh status monitoring
    - Refresh history tracking
    - Orchestrated refresh across multiple monitors
    """
    
    def __init__(
        self,
        monitor_manager: HealthcareMonitorManager,
        workspace_client: Optional[WorkspaceClient] = None
    ):
        """
        Initialize the refresh manager.
        
        Args:
            monitor_manager: HealthcareMonitorManager instance
            workspace_client: Optional WorkspaceClient instance
        """
        self.monitor_manager = monitor_manager
        self.workspace_client = workspace_client or WorkspaceClient()
        
    def refresh_monitor(self, table_name: str) -> Dict:
        """
        Trigger a refresh for a specific monitor.
        
        Args:
            table_name: Full table name
            
        Returns:
            Refresh run information
        """
        print(f"Triggering refresh for {table_name}...")
        
        try:
            run_info = self.workspace_client.quality_monitors.run_refresh(
                table_name=table_name
            )
            
            print(f"✓ Refresh triggered successfully!")
            print(f"  Refresh ID: {run_info.refresh_id}")
            print(f"  State: {run_info.state}")
            
            return {
                "status": "triggered",
                "table_name": table_name,
                "refresh_id": run_info.refresh_id,
                "state": run_info.state
            }
            
        except Exception as e:
            print(f"✗ Error triggering refresh: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_refresh_status(self, table_name: str, refresh_id: str) -> Dict:
        """
        Get the status of a specific refresh run.
        
        Args:
            table_name: Full table name
            refresh_id: Refresh run ID
            
        Returns:
            Refresh status information
        """
        try:
            refresh_info = self.workspace_client.quality_monitors.get_refresh(
                table_name=table_name,
                refresh_id=refresh_id
            )
            
            return {
                "refresh_id": refresh_id,
                "state": refresh_info.state,
                "start_time": refresh_info.start_time_ms,
                "end_time": refresh_info.end_time_ms if hasattr(refresh_info, 'end_time_ms') else None
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def wait_for_refresh(
        self,
        table_name: str,
        refresh_id: str,
        timeout_seconds: int = 1800,
        poll_interval: int = 30
    ) -> Dict:
        """
        Wait for a refresh to complete.
        
        Args:
            table_name: Full table name
            refresh_id: Refresh run ID
            timeout_seconds: Maximum time to wait
            poll_interval: Seconds between status checks
            
        Returns:
            Final refresh status
        """
        print(f"Waiting for refresh {refresh_id} to complete...")
        print(f"  Timeout: {timeout_seconds}s, Poll interval: {poll_interval}s")
        
        start_time = time.time()
        
        while True:
            status = self.get_refresh_status(table_name, refresh_id)
            
            if "error" in status:
                print(f"✗ Error checking status: {status['error']}")
                return status
            
            state = status.get("state", "UNKNOWN")
            print(f"  Current state: {state}")
            
            if state in ["SUCCESS", "COMPLETED"]:
                elapsed = time.time() - start_time
                print(f"✓ Refresh completed successfully in {elapsed:.1f}s")
                return status
            
            elif state in ["FAILED", "CANCELED", "TIMED_OUT"]:
                print(f"✗ Refresh {state.lower()}")
                return status
            
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                print(f"✗ Refresh timed out after {timeout_seconds}s")
                return {"state": "TIMEOUT", "refresh_id": refresh_id}
            
            # Wait before next check
            time.sleep(poll_interval)
    
    def list_refresh_history(self, table_name: str, max_results: int = 10) -> List[Dict]:
        """
        List refresh history for a monitor.
        
        Args:
            table_name: Full table name
            max_results: Maximum number of results to return
            
        Returns:
            List of refresh run information
        """
        try:
            refreshes = self.workspace_client.quality_monitors.list_refreshes(
                table_name=table_name
            )
            
            history = []
            for i, refresh in enumerate(refreshes):
                if i >= max_results:
                    break
                history.append({
                    "refresh_id": refresh.refresh_id,
                    "state": refresh.state,
                    "start_time": refresh.start_time_ms
                })
            
            return history
            
        except Exception as e:
            print(f"Error listing refresh history: {str(e)}")
            return []
    
    def refresh_all_monitors(
        self,
        wait_for_completion: bool = True,
        timeout_seconds: int = 1800
    ) -> Dict[str, Dict]:
        """
        Refresh all healthcare monitors in sequence.
        
        Args:
            wait_for_completion: Whether to wait for each refresh to complete
            timeout_seconds: Timeout for each refresh operation
            
        Returns:
            Dictionary with results for each monitor
        """
        print("=" * 80)
        print("Refreshing All Healthcare Monitors")
        print("=" * 80)
        
        results = {}
        
        # Define all tables to refresh
        tables = [
            ("inference", self.monitor_manager.inference_table),
            ("feature", self.monitor_manager.feature_table),
            ("baseline", self.monitor_manager.baseline_table)
        ]
        
        for name, table_name in tables:
            print(f"\n{'=' * 80}")
            print(f"Refreshing {name.capitalize()} Monitor: {table_name}")
            print(f"{'=' * 80}")
            
            # Trigger refresh
            refresh_result = self.refresh_monitor(table_name)
            
            if refresh_result["status"] == "error":
                results[name] = refresh_result
                continue
            
            # Wait for completion if requested
            if wait_for_completion:
                refresh_id = refresh_result["refresh_id"]
                final_status = self.wait_for_refresh(
                    table_name,
                    refresh_id,
                    timeout_seconds=timeout_seconds
                )
                results[name] = {
                    **refresh_result,
                    "final_state": final_status.get("state", "UNKNOWN")
                }
            else:
                results[name] = refresh_result
        
        print("\n" + "=" * 80)
        print("Refresh Summary")
        print("=" * 80)
        for name, result in results.items():
            if "final_state" in result:
                state = result["final_state"]
                status_symbol = "✓" if state in ["SUCCESS", "COMPLETED"] else "✗"
                print(f"{status_symbol} {name.capitalize()}: {state}")
            else:
                print(f"○ {name.capitalize()}: Triggered (not waited)")
        
        return results
    
    def cancel_refresh(self, table_name: str, refresh_id: str) -> Dict:
        """
        Cancel a running refresh.
        
        Args:
            table_name: Full table name
            refresh_id: Refresh run ID to cancel
            
        Returns:
            Cancellation status
        """
        try:
            self.workspace_client.quality_monitors.cancel_refresh(
                table_name=table_name,
                refresh_id=refresh_id
            )
            print(f"✓ Refresh {refresh_id} canceled")
            return {"status": "canceled", "refresh_id": refresh_id}
        except Exception as e:
            print(f"✗ Error canceling refresh: {str(e)}")
            return {"status": "error", "message": str(e)}


class MonitorAnalyzer:
    """
    Analyze monitoring results and generate insights.
    
    This class provides methods to:
    - Query monitor metric tables
    - Analyze drift patterns
    - Generate summary reports
    - Integrate with custom metrics
    """
    
    def __init__(
        self,
        monitor_manager: HealthcareMonitorManager,
        spark: SparkSession
    ):
        """
        Initialize the monitor analyzer.
        
        Args:
            monitor_manager: HealthcareMonitorManager instance
            spark: Active SparkSession
        """
        self.monitor_manager = monitor_manager
        self.spark = spark
        
    def get_profile_metrics(self, table_name: str, limit: int = 100):
        """
        Get profile metrics from a monitor.
        
        Args:
            table_name: Full table name
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with profile metrics
        """
        try:
            monitor_info = self.monitor_manager.get_monitor_info(table_name)
            profile_table = monitor_info.get("profile_metrics_table_name")
            
            if profile_table:
                return self.spark.table(profile_table).limit(limit)
            else:
                print(f"No profile metrics table found for {table_name}")
                return None
                
        except Exception as e:
            print(f"Error getting profile metrics: {str(e)}")
            return None
    
    def get_drift_metrics(self, table_name: str, limit: int = 100):
        """
        Get drift metrics from a monitor.
        
        Args:
            table_name: Full table name
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with drift metrics
        """
        try:
            monitor_info = self.monitor_manager.get_monitor_info(table_name)
            drift_table = monitor_info.get("drift_metrics_table_name")
            
            if drift_table:
                return self.spark.table(drift_table).limit(limit)
            else:
                print(f"No drift metrics table found for {table_name}")
                return None
                
        except Exception as e:
            print(f"Error getting drift metrics: {str(e)}")
            return None
    
    def generate_monitoring_summary(self) -> Dict:
        """
        Generate a comprehensive monitoring summary.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "monitors": {}
        }
        
        # Check each monitor
        tables = {
            "inference": self.monitor_manager.inference_table,
            "feature": self.monitor_manager.feature_table,
            "baseline": self.monitor_manager.baseline_table
        }
        
        for name, table_name in tables.items():
            monitor_info = self.monitor_manager.get_monitor_info(table_name)
            
            if "error" not in monitor_info:
                summary["monitors"][name] = {
                    "table": table_name,
                    "status": monitor_info.get("status", "unknown"),
                    "has_profile_metrics": monitor_info.get("profile_metrics_table_name") is not None,
                    "has_drift_metrics": monitor_info.get("drift_metrics_table_name") is not None
                }
            else:
                summary["monitors"][name] = {
                    "table": table_name,
                    "status": "error",
                    "error": monitor_info["error"]
                }
        
        return summary

