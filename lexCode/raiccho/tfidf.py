#!/usr/bin/env python
# This function can be executed as-is using the Python2 command "execfile('tfidf.py')" The CCR for the test set will be reported.

import numpy as np
import preProcess as pp
reload(pp)

# Parameters
excludeSimpleWords = 1 # stop words
numList = [100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,12000,14000,16000,18000,20000,25000,30000,35000,40000,0]
isValidate = 1 # If set to 1, it will return CCR after making prediction
typeFlag = 0 # ignore.
doNN = 0 # ignore
kNN = 20 # ignore

dataFile = open('ml_new_train.csv','r')
testFile = open('ml_new_valid.csv','r')
done = 0 #ignore
numList = 0 #numList = 0 will use all available training data.
resFile = open('/home/x/ccrList.txt','w')
for numberTraining in numList:
	# Open files...
	done = 0 #ignore
	dataFile.close()
	testFile.close()
	dataFile = open('ml_new_train.csv','r')
	testFile = open('ml_new_valid.csv','r')
	
	try:
		if done == 1:
			print 'Skipping the word count...'
		else:
			classWordCount, classWordTotal, classTotal, textsWithWord, ftd, fnn, datClass = pp.getWordCount( dataFile, excludeSimpleWords, numberTraining )
			done = 1	
	except NameError:
		classWordCount, classWordTotal, classTotal, textsWithWord, ftd, fnn, datClass = pp.getWordCount( dataFile, excludeSimpleWords, numberTraining )
		done = 1


	print 'Computing TF-IDF!'
	tf, idf, tfidf = pp.getTfIdf( classTotal, textsWithWord, ftd, typeFlag )

	print 'Predicting!'
	predClass, classList, CCR = pp.doTfIdfPrediction( testFile, tfidf, idf, isValidate)

	print numberTraining, CCR
	resFile.write(str(numberTraining) + ',' + str(CCR) + '\n')
resFile.close()

