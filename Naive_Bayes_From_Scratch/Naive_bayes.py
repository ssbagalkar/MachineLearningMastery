# This script implements machine learning algorithm Naive Bayes from scratch

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
#print('Loaded data file {0} with {1} rows').format(filename, len(dataset))



def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

#dataset = [[1], [2], [3], [4], [5]]
splitRatio = 0.67
train, test = splitDataset(dataset, splitRatio)
#print('Split {0} rows into train with {1} and test with {2}').format(len(dataset), train, test)