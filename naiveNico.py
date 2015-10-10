#!/usr/bin/env python

"""
This file implements the Naive Bayes algorithm on a given dataset.
"""

import numpy as np
import string, re

from dataUtils import *

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
@see - (http://stackoverflow.com/a/266162/5025824)
"""
def getImportantWords(text):
    text = re.sub(r'[0-9]+', '', text.translate(None, string.punctuation)) # remove punctuation and numbers
    text = text.lower().decode("utf-8-sig") # lower case and decode to unicode
    text = map(lambda w: ps.stem(w), word_tokenize(text))
    #text = map(lambda w: lemmatizer.lemmatize(w), word_tokenize(text))
    return filter(lambda w: w not in EXCLUDE, text)

"""
Calculate the word probability for each class.
@param data - the data to train with. example of data: [
    (id1, text1, class#),
    (id2, text2, class#),
    (id3, text3, class#)
]
@return - 3 dictionaries:
    - classWordProba - key:class and value:dictionary of word probability.
    - classProba - key:class and value:probability of that class.
    - wordProba - key:word and value:probability of this word showing up.

example of classWordProba: {
    AUTHOR: {word:proba, word:proba, word:proba, ...},
    MOVIE: {word:proba, word:proba, word:proba, ...},
    MUSIC: {word:proba, word:proba, word:proba, ...},
    INTERVIEW: {word:proba, word:proba, word:proba, ...},
}
"""
def learn(data):
    classProba = { # number of examples in a given class / number of examples.
        AUTHOR: 0,
        MOVIE: 0,
        MUSIC: 0,
        INTERVIEW: 0
    }
    classWordProba = { # number of occurences of a specific word in a given class / #of words in the same class.
        AUTHOR: {'TOTAL':0,},
        MOVIE: {'TOTAL':0,},
        MUSIC: {'TOTAL':0,},
        INTERVIEW: {'TOTAL':0,}
    }
    wordProba = {'TOTAL':0,} # number of occurences of each word.

    for item in data:
        classProba[item[2]] = classProba[item[2]]+1.0 # update the counter for that class
        for w in getImportantWords(item[1]): # for each word:
            wordProba[w] = wordProba[w]+1.0 if w in wordProba else 1.0
            wordProba['TOTAL'] = wordProba['TOTAL']+1.0
            # increment the counter of this word for that class.
            classWordProba[item[2]][w] = classWordProba[item[2]][w]+1.0 if w in classWordProba[item[2]] else 1.0
            # increment the counter of words for that class
            classWordProba[item[2]]['TOTAL'] = classWordProba[item[2]]['TOTAL']+1.0

    for key in classWordProba:
        # divide class counter by total number of examples to make it a proba.
        classProba[key] = classProba[key] / len(data)
        for word in classWordProba[key]:
            if word != 'TOTAL':
                # proba of a word (w) given a class (A) = #of time w occurs in a text of class A / #of words in class A
                classWordProba[key][word] = classWordProba[key][word] / classWordProba[key]['TOTAL']

    for w in wordProba:
        if w != 'TOTAL':
            wordProba[w] = wordProba[w]/wordProba['TOTAL']

    return classWordProba, classProba, wordProba



"""
Perform Naive Bayes learning on a training data and produce predictions for a testing data.
@param training_data - the data to train on.
@param testing_data - the data to predict on.
@return - a 2D array with row = item prediciton.
"""
def naiveBayes(training_data, testing_data):
    print "training size: %d" % len(training_data)
    print "testing size: %d" % len(testing_data)

    print "Learninging the data..."
    classWordProba, classProba, wordProba = learn(training_data)
    #print classWordProba
    #print ""
    #print classProba
    #print ""
    #print wordProba
    print "done learning."
    print "Making predictions..."

    predictions = []
    correct = 0
    for item in testing_data:
        #print item

        # P(C|W) = log of probability of the class + sum of the log over all words in W : log(P(C)) + SUM[ log(P(w|C)) - log(P(w)) ]
        author_proba = np.log(classProba[AUTHOR]) # P(author|W)
        movie_proba = np.log(classProba[MOVIE]) # P(movie|W)
        music_proba = np.log(classProba[MUSIC]) # P(music|W)
        interview_proba = np.log(classProba[INTERVIEW]) # P(interview|W)

        for w in getImportantWords(item[1]): # for each word in the test text:
            # update the proba of each class according to this word.
            
            ## AUTHOR ##
            if w in classWordProba[AUTHOR]:
                author_proba = author_proba+np.log(classWordProba[AUTHOR][w])-np.log(wordProba[w])
            else:
                author_proba = author_proba+np.log((1.0/(len(classWordProba[AUTHOR]))))#-np.log((1.0/(len(wordProba))))
            ## MOVIE ##
            if w in classWordProba[MOVIE]:
                movie_proba = movie_proba+np.log(classWordProba[MOVIE][w])-np.log(wordProba[w])
            else:
                movie_proba = movie_proba+np.log((1.0/(len(classWordProba[MOVIE]))))#-np.log((1.0/(len(wordProba))))
            ## MUSIC ##
            if w in classWordProba[MUSIC]:
                music_proba = music_proba+np.log(classWordProba[MUSIC][w])-np.log(wordProba[w])
            else:
                music_proba = music_proba+np.log((1.0/(len(classWordProba[MUSIC]))))#-np.log((1.0/(len(wordProba))))
            ## INTERVIEW ##
            if w in classWordProba[INTERVIEW]:
                interview_proba = interview_proba+np.log(classWordProba[INTERVIEW][w])-np.log(wordProba[w])
            else:
                interview_proba = interview_proba+np.log((1.0/(len(classWordProba[INTERVIEW]))))#-np.log((1.0/(len(wordProba))))

        maxi_proba = max([author_proba, movie_proba, music_proba, interview_proba])

        if author_proba == maxi_proba:
            #print "AUTHOR - 0"
            #predictions.append([item[0], AUTHOR])
            if item[2] == AUTHOR:
                correct = correct+1.0
        elif movie_proba == maxi_proba:
            #print "MOVIE - 1"
            #predictions.append([item[0], MOVIE])
            if item[2] == MOVIE:
                correct = correct+1.0
        elif music_proba == maxi_proba:
            #print "MUSIC - 2"
            #predictions.append([item[0], MUSIC])
            if item[2] == MUSIC:
                correct = correct+1.0
        elif interview_proba == maxi_proba:
            #print "INTERVIEW - 3"
            #predictions.append([item[0], INTERVIEW])
            if item[2] == INTERVIEW:
                correct = correct+1.0
        else:
            print "ERROR: float comparaison."

        #print author_proba
        #print movie_proba
        #print music_proba
        #print interview_proba
        #print ""

    print correct / len(testing_data)
    print "done making predictions."
    return predictions


"""
Uncomment to test:
"""
training_data = readData(PATH_TO_TRAIN_DATA)#, returnSize=10) #max = 53245
data = naiveBayes(training_data[:40000], training_data[40000:])

"""
Uncomment to create the prediction file:
"""
#training_data = readData(PATH_TO_TRAIN_DATA)
#testing_data = readData(PATH_TO_TEST_DATA, isTraining=False)
#data = naiveBayes(training_data, testing_data)
#print "Creating the file..."
#writeData(data)
#print "file created."
