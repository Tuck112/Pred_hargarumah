import streamlit as st
import findspark
import os

# Initialize findspark and Spark
findspark.init()
import pyspark
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, mean as _mean
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.stat import Correlation

# Initialize Spark
spark = SparkSession.builder.appName('DataFrame').getOrCreate()

# Load the data
df_housing = spark.read.option('header', 'true').csv("housing.csv", inferSchema=True)

# Define a function to show DataFrame schema and description
def show_data_info(df):
    st.write("Schema:")
    st.write(df.printSchema())
    st.write("Description (numeric columns):")
    st.write(df.select(['median_house_value', 'median_income', 'total_rooms', 'total_bedrooms', 'population', 'housing_median_age']).describe().toPandas())

# Check for missing values
def check_missing_values(df):
    missing_values = {column: df.filter(col(column).cast("float").isNull()).count() for column in df.columns}
    return missing_values

# Show correlation heatmap
def show_correlation_heatmap(df):
    numeric_columns = [col for col in df.columns if col != 'ocean_proximity']
    assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features")
    df_vector = assembler.transform(df).select("features")
    matrix = Correlation.corr(df_vector, 'features')
    corrmatrix = matrix.collect()[0][0].toArray()
    df_corr = pd.DataFrame(corrmatrix, columns=numeric_columns, index=numeric_columns)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix")
    st.pyplot(plt)

# Train a linear regression model
def train_linear_regression(df):
    df = df.withColumnRenamed('median_house_value', 'price')
    df = df.na.drop()
    mean_bedrooms = df.select(_mean('total_bedrooms')).collect()[0][0]
    df = df.na.fill({'total_bedrooms': mean_bedrooms})
    df = df.withColumn('per_capital_income', df['median_income'] * 10000 / df['population'])
    
    indexer = StringIndexer(inputCol="ocean_proximity", outputCol="ocean_proximity_encoded")
    df = indexer.fit(df).transform(df)
    
    vectorAssembler = VectorAssembler(inputCols=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'price', 'per_capital_income'], outputCol='features')
    vhouse_df = vectorAssembler.transform(df).select(['features', 'price'])
    
    splits = vhouse_df.randomSplit([0.7, 0.3])
    train_df = splits[0]
    test_df = splits[1]
    
    lr = LinearRegression(featuresCol='features', labelCol='price', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train_df)
    acc_train = lr_model.summary.r2
    acc_train2 = lr_model.summary.rootMeanSquaredError
    
    lr_predictions = lr_model.transform(test_df)
    lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="r2")
    acc = lr_evaluator.evaluate(lr_predictions)
    
    return acc_train, acc, lr_predictions

# Streamlit app
st.title("PySpark Housing Data Analysis")

st.header("Data Information")
show_data_info(df_housing)

st.header("Missing Values")
missing_values = check_missing_values(df_housing)
st.write(missing_values)

st.header("Correlation Heatmap")
show_correlation_heatmap(df_housing)

st.header("Linear Regression Model")
acc_train, acc_test, lr_predictions = train_linear_regression(df_housing)
st.write(f"Akurasi data training: {round(acc_train*100, 2)}%")
st.write(f"Akurasi data testing: {round(acc_test*100, 2)}%")

st.header("Predictions")
st.write(lr_predictions.select("prediction", "price", "features").show(5))
