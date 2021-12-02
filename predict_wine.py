# Jung Hyun Kim
# 11/30/2021
# CS 643 Project 2
# predict application

import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier 
from pyspark.ml.classification import RandomForestClassificationModel

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

import sys 



# Create an instance of spark class
spark = SparkSession.builder.appName("WinePredict_Jung_CS643").getOrCreate()

# Fetch the data to be tested from command line argument
predict_csv_file = sys.argv[1]


# Create a Dataframe from the test data set
df_predict = spark.read.format("csv").load(predict_csv_file, inferSchema = True, header = True, sep =";")

# Load the model created from training data sets
rf_model = RandomForestClassificationModel.load("wine_model")


# Predict the test data set
predict_assembler = VectorAssembler(
    inputCols=['"""fixed acidity""""', 
    '""""volatile acidity""""',
    '""""citric acid""""',
    '""""residual sugar""""',
    '""""chlorides""""',
    '""""free sulfur dioxide""""',
    '""""total sulfur dioxide""""',
    '""""density""""',
    '""""pH""""',
    '""""sulphates""""',
    '""""alcohol""""'], 
    outputCol='features')

predict_vector = predict_assembler.transform(df_predict)


predict_data = predict_vector.select('features', '""""quality"""""')



# Prediction
print("\n\n")
print("\n\n")
prediction_raw = rf_model.transform(predict_data)
prediction = prediction_raw.withColumnRenamed('""""quality"""""', 'quality')
prediction.show(20)


# F1 score
pred_and_label = prediction.select("prediction", "quality")

evaluatorMulti = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")

f1 = evaluatorMulti.evaluate(pred_and_label, {evaluatorMulti.metricName: "accuracy"})
acc = evaluatorMulti.evaluate(pred_and_label, {evaluatorMulti.metricName: "f1"})
weighted_precision = evaluatorMulti.evaluate(pred_and_label, {evaluatorMulti.metricName: "weightedPrecision"})
weighted_recall = evaluatorMulti.evaluate(pred_and_label, {evaluatorMulti.metricName: "weightedRecall"})


print("\n\n")
print("==================================================================================================")
print("F1 score : ", f1)
print("==================================================================================================")

print("\n\n")
print("==================================================================================================")
print("Accuracy : ", acc)
print("==================================================================================================")


print("\n\n")
print("==================================================================================================")
print("Weighted Precision : ", weighted_precision)
print("==================================================================================================")


print("\n\n")
print("==================================================================================================")
print("Weighted Recall : ", weighted_recall)
print("==================================================================================================")
print("\n\n")


print("Predict application terminating...")
print("\n\n")