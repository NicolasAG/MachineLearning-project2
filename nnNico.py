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
from sklearn.feature_extraction.text import TfidfVectorizer

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
@param neighbor - the second item to compare (from training_data_data).
@return the distance (always positive) between the two items.
"""
def distance(item, neighbor):
    #ONLY does TF-IDF, find others..?
    text1 = " ".join(getImportantWords(item[1]))
    text2 = " ".join(getImportantWords(neighbor[1]))
    tfidf = TfidfVectorizer().fit_transform([text1, text2])
    return abs( (tfidf * tfidf.T).A[1][0] )
    #return 0

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

    #documents = []
    #similarities = {}
    #for item in training_data:
    #    documents.append(item[1])
    #    similarities[int(item[0])] = 0.0

    i=0
    for item in testing_data:
        print "%d / %d" % (i, len(testing_data))
        i+=1
        #if len(documents) == len(training_data)+1: # if documents contains the previous test-text, remove it.
        #    documents = documents[:-1]
        #documents.append(item[1]) # append the new test-text to the corpus.
        #tfidf = TfidfVectorizer().fit_transform(documents) # learn the tf-idf on the corpus.
        #sim = (tfidf * tfidf.T).A[-1] # take the similarity vector for the last added document: ie: item[1]
        #for i in range(len(training_data)):
        #    similarities[i] = sim[i]

        #####################
    	## Find k nearests ##
        #####################
    	# cf: http://stackoverflow.com/questions/24112259/finding-k-closest-numbers-to-a-given-number
    	#     https://docs.python.org/2/library/heapq.html
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

start = datetime.now()
"""
Uncomment to test:
"""
training_data = readData(PATH_TO_TRAIN_DATA, returnSize=1000) #max = 53245
predictions = kNN(training_data[:800], training_data[800:], k=10, option=1, pLambda=1)

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
