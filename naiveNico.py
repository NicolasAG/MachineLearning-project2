#!/usr/bin/env python

from dataUtils import *
from nltk.corpus import stopwords
import string

PATH_TO_TRAIN_DATA = "./data/ml_dataset_train.csv"
PATH_TO_TEST_DATA = "./data/ml_dataset_test_in.csv"

AUTHOR = '0'
MOVIE = '1'
MUSIC = '2'
INTERVIEWS = '3'

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
@return - dictionary with key:class and value: dictionary of word:probability
example of wordProba: {
    AUTHOR: {word:proba, word:proba, word:proba, ...},
    MOVIE: {word:proba, word:proba, word:proba, ...},
    MUSIC: {word:proba, word:proba, word:proba, ...},
    INTERVIEWS: {word:proba, word:proba, word:proba, ...},
}
"""
def learn(data):
    wordCount = {} # number of occurences of a word regardless of the type of interview.
    wordProba = {
        AUTHOR: {},
        MOVIE: {},
        MUSIC: {},
        INTERVIEWS: {}
    }

    for item in data:
        for w in getImportantWords(item[1]): # for each word:
            # increment it's counter
            wordProba[item[2]][w] = wordProba[item[2]][w]+1.0 if w in wordProba[item[2]] else 1.0
            wordCount[w] = wordCount[w]+1 if w in wordCount else 1

    # proba of a word (w) given a class (A) = #of time w occurs in a text of class A / #of time w occurs in all classes
    for key in wordProba:
        for word in wordProba[key]:
            wordProba[key][word] = wordProba[key][word] / wordCount[word]

    return wordProba



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
    wordProba = learn(training_data)
    print "done learning."
    print "Making predictions..."

    predictions = []
    correct = 0
    for item in testing_data:
        #print item

        author_proba = 1
        movie_proba = 1
        music_proba = 1
        interview_proba = 1

        for w in getImportantWords(item[1]): # for each word in the test text:
            # update the proba of item being in each category.
            author_proba = author_proba*wordProba[AUTHOR][w] if w in wordProba[AUTHOR] else movie_proba*(1.0/len(wordProba[AUTHOR])+1)
            movie_proba = movie_proba*wordProba[MOVIE][w] if w in wordProba[MOVIE] else movie_proba*(1.0/len(wordProba[MOVIE])+1)
            music_proba = music_proba*wordProba[MUSIC][w] if w in wordProba[MUSIC] else movie_proba*(1.0/len(wordProba[MUSIC])+1)
            interview_proba = interview_proba*wordProba[INTERVIEWS][w] if w in wordProba[INTERVIEWS] else movie_proba*(1.0/len(wordProba[INTERVIEWS])+1)

        maxi_proba = max([author_proba, movie_proba, music_proba, interview_proba])

        if author_proba == maxi_proba:
            #predictions.append([item[0], AUTHOR])
            if item[2] == AUTHOR:
                correct = correct+1.0
        elif movie_proba == maxi_proba:
            #predictions.append([item[0], MOVIE])
            if item[2] == MOVIE:
                correct = correct+1.0
        elif music_proba == maxi_proba:
            #predictions.append([item[0], MUSIC])
            if item[2] == MUSIC:
                correct = correct+1.0
        elif interview_proba == maxi_proba:
            #predictions.append([item[0], INTERVIEWS])
            if item[2] == INTERVIEWS:
                correct = correct+1.0
        else:
            print "ERROR: float comparaison."

    #print correct / len(testing_data)
    print "done making predictions."
    return predictions


"""
Uncomment to test:
"""
training_data = readData(PATH_TO_TRAIN_DATA) #53245
data = naiveBayes(training_data[:40000], training_data[40000:])

"""
Uncomment to create the prediction file:
"""
#testing_data = readData(PATH_TO_TEST_DATA, isTraining=False)
#data = naiveBayes(training_data, testing_data)
#print "Creating the file..."
#writeData(data)
#print "file created."
