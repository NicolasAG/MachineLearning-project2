#!/usr/bin/env python

"""
This file implements the k-Nearest Neighbor algorithm on a given dataset.
"""

from dataUtils import *

import numpy as np
import string, re, heapq

from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
#from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
#lemmatizer = WordNetLemmatizer()

PATH_TO_TRAIN_DATA = "./data/ml_dataset_train.csv"
PATH_TO_TEST_DATA = "./data/ml_dataset_test_in.csv"

AUTHOR = '0'
MOVIE = '1'
MUSIC = '2'
INTERVIEW = '3'

EXCLUDE = stopwords.words('english')
EXCLUDE.append('eos')

"""
Return an array of 'important' lowercase word without punctuation given a text.
@param text - the text to take the words from.
@return - an array of words.
@see - http://stackoverflow.com/a/266162/5025824
"""
def getImportantWords(text):
    text = re.sub(r'[0-9]+', '', text.translate(None, string.punctuation)) # remove punctuation and numbers
    text = text.lower().decode("utf-8-sig") # lower case and decode to unicode
    text = map(lambda w: ps.stem(w), word_tokenize(text))
    #text = map(lambda w: lemmatizer.lemmatize(w), word_tokenize(text))
    return filter(lambda w: w not in EXCLUDE, text)

"""
Calculates the 'distance' between two data points.
In this case, one datapoint is a tuple of this from: (id, text, class).
@param item - the first item to compare (from testing_data).
@param neighbor_id - the string id of that neighbor.
@param neighbor_word_proba - the probability of each word for this neighbor.
@return the distance (always positive) between the two items.
"""
item_distances = {} # cache variable that holds all distances from one item to all training points.
def distance(item, neighbor_id, neighbor_word_proba):
    diff = 0.0

    item_word_proba = {} # probability of each word in this item.
    total_words = 0.0 # total number of words in this item.
    for w in getImportantWords(item[1]): # for each word:
        item_word_proba[w] = item_word_proba[w]+1.0 if w in item_word_proba else 1.0
        total_words += 1.0
    for word in item_word_proba:
        item_word_proba[word] /= total_words


    for w1 in item_word_proba: # for each words in query item,
        if w1 in neighbor_word_proba: # if the neighbor has it, add the difference in the proba.
            diff += abs(item_word_proba[w1]-neighbor_word_proba[w1])
        else:
            diff += item_word_proba[w1] #if the neighbor doesn't have it, add the item proba.
    
    for w2 in neighbor_word_proba: # for each word in the neighbor,
        if w2 not in item_word_proba:
            diff += neighbor_word_proba[w2] # if the query doesn't have it, add the neighbor proba.
        #if the query has the word, the difference in the proba was already added in the 1st loop.

    global item_distances # get the cache variable
    item_distances[neighbor_id] = diff # save in the cache.
    return diff

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
@param neighbors_word_proba - the probability of each word for all neighbors.
@return the predicted class of the item.
"""
def getWeightPrediction(item, nn, neighbors_word_proba):
    global item_distances

    prediction = 0.0
    sum_of_weights = 0.0

    for neighbor in nn:
        weight = 0.0
        if neighbor[0] in item_distances: # get the cached distances.
            #d = item_distances[neighbor[0]]
            d = (item_distances[neighbor[0]]**2)
            if d == 0.0:
                print "  WARNING: distance is 0. weight = 1/0.0001"
            weight = 1.0 / (d+0.0001)
        else:
            print "  WARNING: distance not cached. recalculating..."
            #d = distance(item, neighbor[0], neighbors_word_proba[neighbor[0]])
            d = (distance(item, neighbor[0], neighbors_word_proba[neighbor[0]])**2)
            if d == 0.0:
                print "  WARNING: distance is 0. weight = 1/0.0001"
            weight = 1.0 / (d+0.0001)
        prediction += int(neighbor[2]) * weight
        sum_of_weights += weight
    
    if round(prediction/sum_of_weights) < 0.0 or round(prediction/sum_of_weights) > 3.0:
        raise ValueError("ERROR: predicion out of range")

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
def kNN(training_data, testing_data, k=5, option=1):
    print "k = %d" % k
    print "option = %d" % option
    print "training size: %d" % len(training_data)
    print "testing size: %d" % len(testing_data)

    if (k < 1) or (k > len(training_data)):
    	raise ValueError("k must be between 1 and %d" % len(training_data))
    if option not in [1,2,3]:
        raise ValueError("option must be equal to 1 or 2 or 3.")

    print "Making predictions..."
    predictions = [] # array of predictions to return.
    correct = 0 # number of correct predictions.
    training_word_proba = {} # number of occurences of each words in each examples.

    ## Populating training_word_proba ##
    for item in training_data: # for each training example,
        training_word_proba[item[0]] = {} # create a dictionary of word cuont.
        total_words = 0.0 # total number of words in this example.
        for w in getImportantWords(item[1]): # for each word:
            training_word_proba[item[0]][w] = training_word_proba[item[0]][w]+1.0 if w in training_word_proba[item[0]] else 1.0
            total_words += 1.0
        for word in training_word_proba[item[0]]:
            training_word_proba[item[0]][word] /= total_words

    # turning the iterable into an actual heap for better performance. cf: https://docs.python.org/2/library/heapq.html
    heapq.heapify(training_data)

    iteration=1
    for item in testing_data:
        print " %d / %d" % (iteration, len(testing_data))
        iteration+=1
        #####################
        ## Find k nearests ##
        #cf: https://docs.python.org/2/library/heapq.html
        #####################
        global item_distances # get the cache variable.
        item_distances = {} # reset the cached distances.
        nn = heapq.nsmallest(k, training_data, key=lambda x: distance(item, x[0], training_word_proba[x[0]]))
        # At this point, all distances for this test item should be cached.
        #print item_distances
        
        #####################
        ## Make prediction ##
        #####################
        prediction = None
        # OPTION1: average classes with equal weight && predict most popular.
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
        # OPTION2: average classes with equal weight && predict a random class with proba = the average.
        elif option == 2:
            author_proba, movie_proba, music_proba, interview_proba = getEqWeightAvg(nn)
            prediction = np.random.choice(4, p=[author_proba,movie_proba,music_proba,interview_proba])
        # OPTION3: distance-weighted (kernel-based) Nearest Neighbor: predict with weights relative to distance.
        elif option == 3:
            prediction = getWeightPrediction(item, nn, training_word_proba)
        
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
    return predictions, correct/len(testing_data)

start = datetime.now()
"""
Uncomment to test:
"""
CCR = 0.0
for _ in range(10):
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    training_data = readData(PATH_TO_TRAIN_DATA, returnSize=1000) #max = 53245
    predictions, ccr = kNN(training_data[:800], training_data[800:], k=56, option=1)
    CCR += ccr
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print CCR / 10.0


"""
Uncomment to create the prediction file:
"""
#training_data = readData(PATH_TO_TRAIN_DATA)
#testing_data = readData(PATH_TO_TEST_DATA, isTraining=False)
#predictions = knn(training_data, testing_data, k=5)
#print "Creating the file..."
#writeData(predictions)
#print "file created."

print datetime.now() - start
