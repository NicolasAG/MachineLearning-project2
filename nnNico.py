#!/usr/bin/env python

"""
This file implements the k-Nearest Neighbor algorithm on a given dataset.
"""

from dataUtils import *

import numpy as np
import heapq

PATH_TO_TRAIN_DATA = "./data/ml_dataset_train.csv"
PATH_TO_TEST_DATA = "./data/ml_dataset_test_in.csv"

AUTHOR = '0'
MOVIE = '1'
MUSIC = '2'
INTERVIEW = '3'

"""
Calculates the 'distance' between two data points.
In this case, one datapoint is a tuple of this from: (id, text, class).
@param item - the first item to compare (from testing_data).
@param neighbor - the second item to compare (from training_data_data).
@return the distance (always positive) between the two items.
@see: https://www.google.com/search?hl=en&q=python%20tools%20to%20find%20%22distance%22%20between%20two%20texts
@see: http://stackoverflow.com/questions/8897593/similarity-between-two-text-documents
"""
def distance(item, neighbor):
    position1 = 0 #TODO: find features for this text
    position2 = 0 #TODO: find features for this text

    text1 = item[1]
    text2 = neighbor[1]

    return abs(position2 - position1)

"""
Calculates the probability of being in each class given a set of neighbors.
@param nn - array of nearest neighbors of the form: [(id, text, class),(...),...]
@return - 4 probabilities, each one of them corresponding to a different class.
"""
def getEqWeightAvg(nn):
    probas = [0.0, 0.0, 0.0, 0.0]
    for neighbor in nn:
        probas[int(neighbor[2])] += 1
    probas = map(lambda x: x/len(nn), probas)
    return probas[0], probas[1], probas[2], probas[3]

"""
Calculates the prediction of a given item relative to neighbors distance.
@param item - the item to predict for.
@param nn - array of nearest neighbors of the form: [(id, text, class),(...),...]
@param pLambda - used for calculating the weight of a neighbor.
@return the predicted class of the item.
"""
def getWeightPrediction(item, nn, pLambda):
    prediction = 0.0
    sum_of_weights = 0.0

    for neighbor in nn:
        weight = np.exp( - (distance(item, neighbor)**2) / (pLambda**2) )
        prediction += int(neighbor[2]) * weight
        sum_of_weights += weight

    return round(prediction/sum_of_weights)


"""
Classifies each items in testing_data according to the k nearest neighbors.
@param training_data - the data to train on.
@param testing_data - the data to predict on.
@param k - the number of neighbors to look at. (default is 5)
@param option - the variant of the NN algorithm to pick. (default is 1)
    OPTION1: average classes with equal weight && predict most popular.
    OPTION2: average classes with equal weight && predict a random class with proba = the average.
    OPTION3: distance-weighted (kernel-based) Nearest Neighbor: predict with weights relative to distance.
@param pLambda - used for calculating the weight of a neighbor (only relevant for OPTION3).
@return - a 2D array with row = item prediciton.
"""
def kNN(training_data, testing_data, k=5, option=1, pLambda=1):
    print "training size: %d" % len(training_data)
    print "testing size: %d" % len(testing_data)

    if (k < 1) or (k > len(training_data)):
    	raise ValueError("k must be between 1 and %d" % len(training_data))
    if option not in [1,2,3]:
        raise ValueError("option must be equal to 1 or 2 or 3.")

    print "Making predictions..."
    predictions = []
    correct = 0
    heapq.heapify(training_data) # turning the iterable into an actual heap for better performance.

    for item in testing_data:
        #####################
    	## Find k nearests ##
        #####################
    	# cf: http://stackoverflow.com/questions/24112259/finding-k-closest-numbers-to-a-given-number
    	#     https://docs.python.org/2/library/heapq.html#heapq.nsmallest
    	nn = heapq.nsmallest(k, training_data, key = lambda x: distance(item, x))

        #####################
        ## Make prediction ##
        #####################
        prediction = None
        # OPTION1: average their class with equal weight && predict most popular.
        if option == 1:
            author_proba, movie_proba, music_proba, interview_proba = getEqWeightAvg(nn)
            maxi_proba = max([author_proba, movie_proba, music_proba, interview_proba])
            if author_proba == maxi_proba:
                prediction = 0
            elif movie_proba == maxi_proba:
                prediction = 1
            elif music_proba == maxi_proba:
                prediction = 2
            elif interview_proba == maxi_proba:
                prediction = 3
        # OPTION2: average their class with equal weight && predict a random class with proba = the average.
        elif option == 2:
            author_proba, movie_proba, music_proba, interview_proba = getEqWeightAvg(nn)
            prediction = np.random.choice(4, p=[author_proba,movie_proba,music_proba,interview_proba])
        # OPTION3: distance-weighted (kernel-based) Nearest Neighbor: predict with weights relative to distance.
        elif option == 3:
            prediction = getWeightPrediction(item, nn, pLambda)
        
        ##############################
        ## Append prediction result ##
        ##############################
        if prediction == 0:
            #print "AUTHOR - 0"
            #predictions.append([item[0], AUTHOR])
            if item[2] == AUTHOR:
                correct += 1.0
        elif prediction == 1:
            #print "MOVIE - 1"
            #predictions.append([item[0], MOVIE])
            if item[2] == MOVIE:
                correct += 1.0
        elif prediction == 2:
            #print "MUSIC - 2"
            #predictions.append([item[0], MUSIC])
            if item[2] == MUSIC:
                correct += 1.0
        elif prediction == 3:
            #print "INTERVIEW - 3"
            #predictions.append([item[0], INTERVIEW])
            if item[2] == INTERVIEW:
                correct += 1.0
        else:
            print "ERROR: no prediction made!"

    print correct / len(testing_data)
    print "done making predictions."
    return predictions

"""
Uncomment to test:
"""
training_data = readData(PATH_TO_TRAIN_DATA)#, returnSize=10) #max = 53245
predictions = kNN(training_data[:40000], training_data[40000:], k=100, option=3, pLambda=1)

"""
Uncomment to create the prediction file:
"""
#training_data = readData(PATH_TO_TRAIN_DATA)
#testing_data = readData(PATH_TO_TEST_DATA, isTraining=False)
#predictions = knn(training_data, testing_data, k=5)
#print "Creating the file..."
#writeData(predictions)
#print "file created."
