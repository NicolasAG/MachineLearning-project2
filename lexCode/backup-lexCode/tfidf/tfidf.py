#!/usr/bin/env python

import numpy as np
import preProcess as pp


excludeSimpleWords = 1
numberTraining = 0
isValidate = 1
typeFlag = 0
doNN = 1
kNN = 20
dataFile = open('ml_new_train.csv','r')
testFile = open('ml_new_valid.csv','r')

try:
	if done == 1:
		print 'Skipping the word count...'
	else:
		classWordCount, classWordTotal, classTotal, textsWithWord, ftd, fnn, datClass = pp.getWordCount( dataFile, excludeSimpleWords, numberTraining )
		done = 1	
except NameError:
	classWordCount, classWordTotal, classTotal, textsWithWord, ftd = pp.getWordCount( dataFile, excludeSimpleWords, numberTraining )
	done = 1

#print ftd[0][0]
print 'Computing TF-IDF!'
tf, idf, tfidf = pp.getTfIdf( classTotal, textsWithWord, ftd, typeFlag )
if doNN == 1:
	#getTfNN( N, nt, fnn, flagType )
	tfNN = pp.getTfNN( fnn )
print 'Predicting!'

#predClass = pp.doTfIdfPrediction( testFile, tfidf, idf, isValidate)
predClass = pp.getTfidfNN( testFile, tfNN, idf, datClass, kNN, isValidate )
#Ndoc = sum(classTotal)
