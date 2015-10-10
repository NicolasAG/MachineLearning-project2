#!/usr/bin/env python

"""
This file implements the k-Nearest Neighbor algorithm on a given dataset.
"""

from dataUtils import *

import numpy as np
from heapq import nsmallest

"""
Calculates the 'distance' between two data points.
In this case, one datapoint is a tuple of this from: (id, text, class).
@param item1 - the first item to compare.
@param item2 - the second item to compare.
@return the distance (always positive) between the two items.
"""
def distance(item1, item2):
	position1 = 0 #TODO: find features for this text
	position2 = 0 #TODO: find features for this text
	
	return abs(position2 - position1)

"""
Classifies each items in testing_data according to the k nearest neighbors.
@param training_data - the data to train on.
@param testing_data - the data to predict on.
@param k - the number of neighbors to look at. (default is 5)
@return - a 2D array with row = item prediciton.
"""
def kNN(training_data, testing_data, k=5):
	print "training size: %d" % len(training_data)
    print "testing size: %d" % len(testing_data)

    predictions = []
    correct = 0
    
    for item in testing_data:
    	#TODO: find k nearest
    	#http://stackoverflow.com/questions/24112259/finding-k-closest-numbers-to-a-given-number
    	nsmallest(k, training_data, key=lambda x: distance(x, item))

    	#OPTION1: average their class with equal weight.
    	#OPTION2: average their class with weights relative to their distance.

    	#OPTION1: predict most popular.
    	#OPTION2: predict a random class with proba = the average.


    return predictions

"""
Uncomment to test:
"""
training_data = readData(PATH_TO_TRAIN_DATA)#, returnSize=10) #max = 53245
predictions = kNN(training_data[:40000], training_data[40000:], k=5)

"""
Uncomment to create the prediction file:
"""
#training_data = readData(PATH_TO_TRAIN_DATA)
#testing_data = readData(PATH_TO_TEST_DATA, isTraining=False)
#predictions = knn(training_data, testing_data, k=5)
#print "Creating the file..."
#writeData(predictions)
#print "file created."
