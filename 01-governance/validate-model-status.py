# Databricks notebook source
"""
Model Governance Validation Script

This script validates the governance status of the healthcare insurance model.
It checks:
- Model version details
- Champion alias assignment
- Governance tags
- Performance metrics
- Compliance status
"""

import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

# Configure Unity Catalog
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

def validate_model_governance(model_name="juan_dev.healthcare_data.insurance_model"):
    """
    Comprehensive validation of model governance status
    """
    print("=" * 80)
    print("HEALTHCARE MODEL GOVERNANCE VALIDATION")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        # 1. Check if champion model exists
        print("\n[1] CHAMPION MODEL STATUS")
        print("-" * 80)
        try:
            champion_info = client.get_model_version_by_alias(model_name, "champion")
            print(f"✓ Champion model found")
            print(f"  Version: {champion_info.version}")
            print(f"  Status: {champion_info.status}")
            print(f"  Run ID: {champion_info.run_id}")
            print(f"  Created: {champion_info.creation_timestamp}")

            champion_version = champion_info.version
            champion_exists = True
        except Exception as e:
            print(f"✗ No champion model found: {e}")
            champion_exists = False
            champion_version = None

        # 2. List all model versions
        print("\n[2] ALL MODEL VERSIONS")
        print("-" * 80)
        versions = client.search_model_versions(f"name = '{model_name}'")

        if not versions:
            print("✗ No model versions found")
            return False

        print(f"Total versions found: {len(versions)}")
        for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
            is_champion = "(CHAMPION)" if champion_exists and v.version == champion_version else ""
            print(f"  Version {v.version} {is_champion}")
            print(f"    Status: {v.status}")
            print(f"    Run ID: {v.run_id}")

        latest_version = max([int(v.version) for v in versions])

        # 3. Check governance tags
        print("\n[3] GOVERNANCE TAGS")
        print("-" * 80)
        if champion_exists:
            tags = champion_info.tags
            if tags:
                print(f"Champion model tags:")
                for key, value in tags.items():
                    print(f"  {key}: {value}")

                # Check required governance tags
                required_tags = ['healthcare_compliance', 'validation_r2']
                missing_tags = [tag for tag in required_tags if tag not in tags]

                if missing_tags:
                    print(f"⚠ Missing governance tags: {missing_tags}")
                else:
                    print(f"✓ All required governance tags present")
            else:
                print("⚠ No tags found on champion model")
        else:
            print("✗ Cannot check tags - no champion model")

        # 4. Retrieve and validate metrics
        print("\n[4] PERFORMANCE METRICS")
        print("-" * 80)
        if champion_exists:
            try:
                run_data = client.get_run(champion_info.run_id)
                metrics = run_data.data.metrics

                print(f"Retrieved metrics from run:")
                for metric_name, metric_value in sorted(metrics.items()):
                    print(f"  {metric_name}: {metric_value:.4f}")

                # Validate against healthcare requirements
                print("\n  Validation against healthcare requirements:")
                requirements = {
                    "min_r2_score": 0.70,
                    "max_mae": 15.0,
                    "min_high_risk_accuracy": 0.60
                }

                r2_score = metrics.get('r2_score', 0)
                mae = metrics.get('mean_absolute_error', float('inf'))
                high_risk_acc = metrics.get('high_risk_accuracy', 0)

                r2_check = r2_score >= requirements['min_r2_score']
                mae_check = mae <= requirements['max_mae']
                accuracy_check = high_risk_acc >= requirements['min_high_risk_accuracy']

                print(f"  R² Score: {r2_score:.4f} >= {requirements['min_r2_score']} → {'✓ PASS' if r2_check else '✗ FAIL'}")
                print(f"  MAE: {mae:.2f} <= {requirements['max_mae']} → {'✓ PASS' if mae_check else '✗ FAIL'}")
                print(f"  High Risk Acc: {high_risk_acc:.4f} >= {requirements['min_high_risk_accuracy']} → {'✓ PASS' if accuracy_check else '✗ FAIL'}")

                all_checks_pass = r2_check and mae_check and accuracy_check

                if all_checks_pass:
                    print("\n  ✓ Champion model meets all healthcare requirements")
                else:
                    print("\n  ✗ Champion model does NOT meet all healthcare requirements")

            except Exception as e:
                print(f"✗ Could not retrieve metrics: {e}")
                all_checks_pass = False
        else:
            print("✗ Cannot check metrics - no champion model")
            all_checks_pass = False

        # 5. Model description validation
        print("\n[5] MODEL DESCRIPTION")
        print("-" * 80)
        if champion_exists:
            description = champion_info.description
            if description:
                # Show first 300 characters
                print(f"Description preview:")
                print(description[:300] + "..." if len(description) > 300 else description)

                # Check for key terms in description
                required_terms = ['Healthcare', 'HIPAA', 'Performance Metrics']
                missing_terms = [term for term in required_terms if term not in description]

                if missing_terms:
                    print(f"\n⚠ Description missing key terms: {missing_terms}")
                else:
                    print(f"\n✓ Description contains all required information")
            else:
                print("⚠ No description found")
        else:
            print("✗ Cannot check description - no champion model")

        # 6. Final summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        checks = {
            "Champion model exists": champion_exists,
            "Governance tags present": champion_exists and len(tags) > 0 if champion_exists else False,
            "Performance requirements met": all_checks_pass,
            "Model description complete": champion_exists and bool(champion_info.description) if champion_exists else False
        }

        for check_name, check_result in checks.items():
            status = "✓ PASS" if check_result else "✗ FAIL"
            print(f"{check_name}: {status}")

        overall_pass = all(checks.values())
        print("\n" + "=" * 80)
        if overall_pass:
            print("✓✓✓ OVERALL STATUS: GOVERNANCE VALIDATION PASSED ✓✓✓")
        else:
            print("✗✗✗ OVERALL STATUS: GOVERNANCE VALIDATION FAILED ✗✗✗")
        print("=" * 80)

        return overall_pass

    except Exception as e:
        print(f"\n✗ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

# COMMAND ----------

# Run validation
validation_result = validate_model_governance()

if validation_result:
    print("\n✓ Model is ready for production deployment")
else:
    print("\n✗ Model requires additional governance steps before deployment")

# COMMAND ----------

# Additional utility: Compare multiple versions
def compare_model_versions(model_name="juan_dev.healthcare_data.insurance_model", num_versions=3):
    """
    Compare performance metrics across recent model versions
    """
    print("\n" + "=" * 80)
    print("MODEL VERSION COMPARISON")
    print("=" * 80)

    try:
        versions = client.search_model_versions(f"name = '{model_name}'")
        versions_sorted = sorted(versions, key=lambda x: int(x.version), reverse=True)[:num_versions]

        print(f"\nComparing top {len(versions_sorted)} versions:\n")

        comparison_data = []
        for v in versions_sorted:
            try:
                run_data = client.get_run(v.run_id)
                metrics = run_data.data.metrics

                version_info = {
                    'version': v.version,
                    'r2_score': metrics.get('r2_score', 0),
                    'mae': metrics.get('mean_absolute_error', 0),
                    'rmse': metrics.get('root_mean_squared_error', 0),
                    'high_risk_acc': metrics.get('high_risk_accuracy', 0),
                    'status': v.status
                }
                comparison_data.append(version_info)

            except Exception as e:
                print(f"Could not get metrics for version {v.version}: {e}")

        # Display comparison table
        if comparison_data:
            print(f"{'Version':<10} {'R²':<10} {'MAE':<10} {'RMSE':<10} {'HR Acc':<10} {'Status':<15}")
            print("-" * 80)
            for data in comparison_data:
                print(f"{data['version']:<10} {data['r2_score']:<10.4f} {data['mae']:<10.2f} "
                      f"{data['rmse']:<10.2f} {data['high_risk_acc']:<10.4f} {data['status']:<15}")

            # Identify best version
            best_version = max(comparison_data, key=lambda x: x['r2_score'])
            print(f"\nBest performing version by R²: Version {best_version['version']} (R²={best_version['r2_score']:.4f})")

        print("=" * 80)

    except Exception as e:
        print(f"Version comparison failed: {e}")

# COMMAND ----------

# Optionally run comparison
# Uncomment to compare versions
# compare_model_versions()
