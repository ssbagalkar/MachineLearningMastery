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
import math

## Write a function to load the csv data
def loadCsv(filename):
    lines = csv.reader(open(filename,"r"))
    dataset = list(lines)
    myList = []
    for ii in range(len(dataset)):
        a = [float(x) for x in dataset[ii][0].split(',')]
        myList.append(a)
    return myList

filename = 'pima-indians.csv'
dataset = loadCsv(filename)

print('Loaded data file {0} with {1} rows'.format(filename, len(dataset)))

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
print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(train), len(test)))

# Separate data by class
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


# define function to calculate  mean
def mean(numbers):
    return sum(numbers)/float(len(numbers))

# define function to calculate standard deviation
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

# define function to summarize dataset
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

# define function to summarize dataset by class
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

summary = summarizeByClass(train)

# calculate probability
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

