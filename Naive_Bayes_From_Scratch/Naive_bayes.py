# This script implements machine learning algorithm Naive Bayes from scratch
# https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
## Here's how we will solve it
#Handle Data: Load the data from CSV file and split it into training and test datasets.
#Summarize Data: summarize the properties in the training dataset so that we can calculate probabilities and make predictions.
#Make a Prediction: Use the summaries of the dataset to generate a single prediction.
#Make Predictions: Generate predictions given a test dataset and a summarized training dataset.
#Evaluate Accuracy: Evaluate the accuracy of predictions made for a test dataset as the percentage correct out of all predictions made.
#Tie it Together: Use all of the code elements to present a complete and standalone implementation of the Naive Bayes algorithm.

import csv
import random

## Write a function to load the csv data
def loadCsv(filename):
    lines = csv.reader(open(filename,"r"))
    dataset = list(lines)
    return dataset

filename = 'pima-indians.csv'
dataset = loadCsv(filename)


# Funtion to split dataset into train and test
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

splitRatio = 0.67
train, test = splitDataset(dataset, splitRatio)

# Separate data by class
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


separated = separateByClass(dataset)

