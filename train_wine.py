# Jung Hyun Kim
# 11/30/2021
# CS 643 Project 2
# train application

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



# Path to training data set
train_csv_file = "TrainingDataset.csv"

# Create an instance of spark class
spark = SparkSession.builder.appName("WinePredict_Jung_CS643").getOrCreate()

# Create a spark dataframe of input csv file
df_train = spark.read.format("csv").load(train_csv_file, inferSchema = True, header = True, sep =";")


# Create vectors from feature columns that determine the wine quality
#   Apache MLlib takes input in vector form
assembler = VectorAssembler(
    inputCols=['"""""fixed acidity""""', 
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

data_vector = assembler.transform(df_train)


training_data = data_vector.select('features', '""""quality"""""')

# Create an object of RandomForestClassifier class
wine_rf = RandomForestClassifier(featuresCol='features', labelCol='""""quality"""""', maxDepth=10)


# Pass in the training data to train the LR model
rf_model = wine_rf.fit(training_data)

# Evaluate the model
result = rf_model.evaluate(training_data)

# transform() is for prediction
prediction = rf_model.transform(training_data).withColumnRenamed('""""quality"""""', 'quality')
#prediction.show(15)


# Calculate metrics

evaluator = MulticlassClassificationEvaluator(labelCol='quality', predictionCol="prediction")
f1 = evaluator.evaluate(prediction, {evaluator.metricName: "f1"})
acc = evaluator.evaluate(prediction, {evaluator.metricName: "accuracy"})
weighted_precision = evaluator.evaluate(prediction, {evaluator.metricName: "weightedPrecision"})
weighted_recall = evaluator.evaluate(prediction, {evaluator.metricName: "weightedRecall"})

print("\n\n")
print("========================================================================")
print("F1 score : ", f1)
print("========================================================================")


print("\n\n")
print("========================================================================")
print("Accuracy : ", acc)
print("========================================================================")


print("\n\n")
print("========================================================================")
print("Weighted Precision: ", weighted_precision)
print("========================================================================")


print("\n\n")
print("========================================================================")
print("Weighted Recall: ", weighted_recall)
print("========================================================================")



# Save the model
#rf_model.save("wine_model")
rf_model.write().overwrite().save("wine_model")


print("\n\n")
print("Train application terminating...")
print("\n\n")