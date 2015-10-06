#!/usr/bin/env python

from dataUtils import *
from nltk.corpus import stopwords
import string

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
    return filter(lambda w: w not in EXCLUDE, text.translate(None, string.punctuation).lower().split())

"""
Calculate the word probability for each class.
@param data - the data to train with. example of data: [
    (id1, text1, class#),
    (id2, text2, class#),
    (id3, text3, class#)
]
@return - 2 dictionaries with key:class and value:
    - dictionary of word probability for the first one.
    - probability of that class for the second one.

example of wordProba: {
    AUTHOR: {word:proba, word:proba, word:proba, ...},
    MOVIE: {word:proba, word:proba, word:proba, ...},
    MUSIC: {word:proba, word:proba, word:proba, ...},
    INTERVIEWS: {word:proba, word:proba, word:proba, ...},
}
"""
def learn(data):
    classProba = { # number of interviews in a given class / number of interviews.
        AUTHOR: 0,
        MOVIE: 0,
        MUSIC: 0,
        INTERVIEW: 0
    }
    wordCount = { # number of words in a given class.
        AUTHOR: 0,
        MOVIE: 0,
        MUSIC: 0,
        INTERVIEW: 0
    }
    wordProba = { # number of occurences of a specific word in a given class / #of words in the same class.
        AUTHOR: {},
        MOVIE: {},
        MUSIC: {},
        INTERVIEW: {}
    }

    for item in data:
        classProba[item[2]] = classProba[item[2]]+1.0
        for w in getImportantWords(item[1]): # for each word:
            # increment it's counter
            wordProba[item[2]][w] = wordProba[item[2]][w]+1.0 if w in wordProba[item[2]] else 1.0
            wordCount[item[2]] = wordCount[item[2]]+1.0

    # proba of a word (w) given a class (A) = #of time w occurs in a text of class A / #of words in class A
    for key in wordProba:
        classProba[key] = classProba[key] / len(data)
        for word in wordProba[key]:
            wordProba[key][word] = wordProba[key][word] / wordCount[key]


    return wordProba, classProba



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
    wordProba, classProba = learn(training_data)
    print "done learning."
    print "Making predictions..."

    predictions = []
    correct = 0
    for item in testing_data:
        #print item

        author_proba = 1 # P(author|W)
        movie_proba = 1 # P(movie|W)
        music_proba = 1 # P(music|W)
        interview_proba = 1 # P(interview|W)

        for w in getImportantWords(item[1]): # for each word in the test text:
            # update the proba of item being in each category:
            # P(C|W) = product over all words in W of : P(w|C)*P(C) / P(w)<-->can be ignored
            author_proba = author_proba*classProba[AUTHOR]*wordProba[AUTHOR][w] if w in wordProba[AUTHOR] else author_proba*classProba[AUTHOR]*(1.0/(len(wordProba[AUTHOR])+1))
            movie_proba = movie_proba*classProba[MOVIE]*wordProba[MOVIE][w] if w in wordProba[MOVIE] else movie_proba*classProba[MOVIE]*(1.0/(len(wordProba[MOVIE])+1))
            music_proba = music_proba*classProba[MUSIC]*wordProba[MUSIC][w] if w in wordProba[MUSIC] else music_proba*classProba[MUSIC]*(1.0/(len(wordProba[MUSIC])+1))
            interview_proba = interview_proba*classProba[INTERVIEW]*wordProba[INTERVIEW][w] if w in wordProba[INTERVIEW] else interview_proba*classProba[INTERVIEW]*(1.0/(len(wordProba[INTERVIEW])+1))

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
training_data = readData(PATH_TO_TRAIN_DATA)#, returnSize=100) #max = 53245
data = naiveBayes(training_data[:40000], training_data[40000:])

"""
Uncomment to create the prediction file:
"""
#testing_data = readData(PATH_TO_TEST_DATA, isTraining=False)
#data = naiveBayes(training_data, testing_data)
#print "Creating the file..."
#writeData(data)
#print "file created."
