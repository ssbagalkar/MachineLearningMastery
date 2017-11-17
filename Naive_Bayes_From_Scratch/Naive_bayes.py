# This script implements machine learning algorithm Naive Bayes from scratch

## Here's how we will solve it
#Handle Data: Load the data from CSV file and split it into training and test datasets.
#Summarize Data: summarize the properties in the training dataset so that we can calculate probabilities and make predictions.
#Make a Prediction: Use the summaries of the dataset to generate a single prediction.
#Make Predictions: Generate predictions given a test dataset and a summarized training dataset.
#Evaluate Accuracy: Evaluate the accuracy of predictions made for a test dataset as the percentage correct out of all predictions made.
#Tie it Together: Use all of the code elements to present a complete and standalone implementation of the Naive Bayes algorithm.

import csv

## Write a function to load the csv data
def loadCsv(filename):
    lines = csv.reader(open(filename,"rb"))
    dataset = list (lines)
    for i in range(len(dataset)):
        dataset[i] = [ float(x) for x in dataset[i]]
    return dataset