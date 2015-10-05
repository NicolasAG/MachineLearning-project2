#!/usr/bin/env python

from readdata import readData
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
Trying to get the word proba for each class.
@param data - the data to train with. example of data: [
    (id1, text1, class#),
    (id2, text2, class#),
    (id3, text3, class#)
]
@return - dictionary of the form: {
	AUTHOR: {word:proba, word:proba, word:proba, ...},
	MOVIE: {word:proba, word:proba, word:proba, ...},
	MUSIC: {word:proba, word:proba, word:proba, ...},
	INTERVIEWS: {word:proba, word:proba, word:proba, ...},
}
"""
def NaiveBayes(data):
	wordCount = {} # #of occurences of a word regardless of the type of interview.
	wordProba = {
		AUTHOR: {},
		MOVIE: {},
		MUSIC: {},
		INTERVIEWS: {}
	}
	'''
	P(w|A) = P(w,A) / P(A)
	proba of a word (w) given a class (A) = #of time w occurs in a text of class A / #of time w occurs in all classes
	'''
	for item in data:
		#lowercase word without punctuation (http://stackoverflow.com/a/266162/5025824)
		words = item[1].translate(None, string.punctuation).lower().split()
		'''
		TODO:
		fix this:
			naiveNico.py:53: UnicodeWarning: Unicode equal comparison failed to convert both arguments to Unicode - interpreting them as being unequal
	  			for w in filter(lambda word: not word in EXCLUDE, words): # for each 'important' word:
		'''
		for w in filter(lambda word: not word in EXCLUDE, words): # for each 'important' word:
			# increment it's counter
			wordProba[item[2]][w] = wordProba[item[2]][w]+1 if w in wordProba[item[2]] else 1
			wordCount[w] = wordCount[w]+1 if w in wordCount else 1

	for key in wordProba:
		for word in wordProba[key]:
			wordProba[key][word] = float(wordProba[key][word]) / float(wordCount[word])

	return wordProba


training_data = readData(PATH_TO_TRAIN_DATA, returnSize=1000)
print NaiveBayes(training_data)


