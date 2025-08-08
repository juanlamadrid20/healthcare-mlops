#!/usr/bin/env python3
"""
Test script to validate the ML-prefixed monitoring system
This validates the simplified healthcare model monitoring setup
"""

def test_ml_monitoring_views():
    """Test that all ML monitoring views are accessible and return data"""
    
    print("🧪 Testing ML Monitoring Views")
    print("=" * 50)
    
    # ML monitoring views to test
    views_to_test = [
        "juan_dev.healthcare_data.ml_drift_monitor",
        "juan_dev.healthcare_data.ml_monitoring_summary", 
        "juan_dev.healthcare_data.ml_model_alerts"
    ]
    
    results = {}
    
    for view_name in views_to_test:
        print(f"\n🔍 Testing: {view_name}")
        
        try:
            # Test basic access
            df = spark.table(view_name)
            row_count = df.count()
            columns = df.columns
            
            print(f"   ✅ Access: SUCCESS")
            print(f"   📊 Rows: {row_count:,}")
            print(f"   📋 Columns: {len(columns)} ({', '.join(columns[:5])}{'...' if len(columns) > 5 else ''})")
            
            # Show sample data if available
            if row_count > 0:
                print(f"   📄 Sample data:")
                df.limit(3).show(truncate=False)
            else:
                print(f"   ⚠️  No data found (may be expected for recent periods)")
            
            results[view_name] = {
                "accessible": True,
                "row_count": row_count,
                "columns": len(columns)
            }
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results[view_name] = {
                "accessible": False, 
                "error": str(e)
            }
    
    print(f"\n📋 Test Summary:")
    print("-" * 30)
    
    accessible_count = sum(1 for r in results.values() if r.get("accessible", False))
    total_count = len(results)
    
    print(f"Views accessible: {accessible_count}/{total_count}")
    
    for view_name, result in results.items():
        status = "✅" if result.get("accessible", False) else "❌"
        short_name = view_name.split(".")[-1]  # Get just the view name
        print(f"   {status} {short_name}")
    
    return results

def test_source_tables():
    """Test that source tables are accessible"""
    
    print("\n🗃️  Testing Source Tables")
    print("=" * 50)
    
    source_tables = [
        "juan_dev.healthcare_data.silver_patients",
        "juan_dev.healthcare_data.ml_patient_predictions"
    ]
    
    for table_name in source_tables:
        try:
            df = spark.table(table_name)
            row_count = df.count()
            print(f"✅ {table_name}: {row_count:,} records")
            
            # Show a few columns to verify schema
            columns = df.columns[:8]  # First 8 columns
            print(f"   Columns: {', '.join(columns)}...")
            
        except Exception as e:
            print(f"❌ {table_name}: Error - {str(e)}")

def run_monitoring_validation():
    """Run complete monitoring validation"""
    
    print("🚀 ML Monitoring System Validation")
    print("🎯 Objective: Validate lakehouse monitoring fundamentals")
    print("🏷️  Asset Naming: ML-prefixed monitoring views")
    print("=" * 70)
    
    # Test 1: Source tables
    test_source_tables()
    
    # Test 2: ML monitoring views
    view_results = test_ml_monitoring_views()
    
    # Test 3: Basic monitoring queries
    print(f"\n🔍 Testing Basic Monitoring Queries")
    print("=" * 50)
    
    try:
        # Test drift monitoring query
        print("Testing drift monitoring query...")
        drift_query = """
        SELECT 
            prediction_date,
            daily_predictions,
            avg_prediction,
            volume_status
        FROM juan_dev.healthcare_data.ml_drift_monitor 
        ORDER BY prediction_date DESC 
        LIMIT 5
        """
        
        drift_df = spark.sql(drift_query)
        drift_count = drift_df.count()
        
        if drift_count > 0:
            print(f"✅ Drift monitoring: {drift_count} recent periods")
            drift_df.show()
        else:
            print("⚠️  Drift monitoring: No recent data (may be expected)")
            
    except Exception as e:
        print(f"❌ Drift monitoring query failed: {e}")
    
    # Overall assessment
    print(f"\n🎉 Validation Complete!")
    print("=" * 50)
    
    accessible_views = sum(1 for r in view_results.values() if r.get("accessible", False))
    total_views = len(view_results)
    
    if accessible_views == total_views:
        print("✅ All ML monitoring views are working correctly!")
        print("🎯 Lakehouse monitoring fundamentals validated successfully")
        print("🚀 Ready to add business logic incrementally")
    else:
        print(f"⚠️  {accessible_views}/{total_views} views working - review issues above")
        print("🔧 Fix any view creation issues before proceeding")
    
    print(f"\n📊 ML Asset Summary:")
    print("   • ml_drift_monitor: Daily prediction metrics and drift detection")
    print("   • ml_monitoring_summary: Overall model performance summary")  
    print("   • ml_model_alerts: Automated alerting for anomalies")
    
    return view_results

if __name__ == "__main__":
    # This would run if executed as a script
    # In Databricks, you would run the functions directly
    print("ML Monitoring Validation Script")
    print("Run: run_monitoring_validation() in Databricks")
    test_ml_monitoring_views()
    run_monitoring_validation()