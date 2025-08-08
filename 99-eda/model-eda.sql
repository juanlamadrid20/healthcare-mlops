-- Databricks notebook source
select * from juan_dev.ml.healthcare_features;

-- COMMAND ----------

delete from juan_dev.ml.healthcare_features;

-- COMMAND ----------

select * from juan_dev.healthcare_data.dim_patients limit 10;

-- COMMAND ----------

select * from juan_dev.healthcare_data.silver_patients limit 10;



-- COMMAND ----------

select * from juan_dev.healthcare_data.ml_insurance_features limit 10;

-- COMMAND ----------


