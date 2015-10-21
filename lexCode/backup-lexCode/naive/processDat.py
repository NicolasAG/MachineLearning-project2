#!/usr/bin/env python

# COMP 598 - Project 2 - Naive Bayes
import preProcess as pp
import numpy as np

# Flags
excludeSimpleWords = 1	# Exclude articles (stopwords in nltk)
isValidate = 1		# If 1, compare predicted and actual classes

# Parameters
numberTraining = 0 	# If >0, training will stop after this number of examples
numClasses = 4		# Number of classes in data. Should probably detect it automatically, but meh.

trainFile = 'ml_new_train.csv'
testFile = 'ml_new_valid.csv'

# Declarations
pc = [0.0 for i in range(numClasses)]	# Probability of a class, P(c)
pwc = [{} for i in range(numClasses)]	# Probability of a word given a class, P(w|c)

#with open('ml_dataset_train.csv', 'r') as dataFile:
with open(trainFile,'r') as dataFile:

	# Count words
	classWordCount, classWordTotal, classTotal = pp.getWordCount( dataFile, excludeSimpleWords, numberTraining )

	# Get probabilities
	Pc, Pwc = pp.getProb( classWordCount, classWordTotal, classTotal )
	# p( c )

	# Get prediction on validation.
	predClass, CCR = pp.doPrediction( open(testFile,'r'), Pc, Pwc, classWordTotal, isValidate )
	print CCR
