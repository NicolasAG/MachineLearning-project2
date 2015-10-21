#!/usr/bin/env python

import numpy as np
import preProcess as pp
reload(pp)

excludeSimpleWords = 1
numberTraining = 300
numList = [100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,12000,14000,16000,18000,20000,25000,30000,35000,40000,0]
isValidate = 1
typeFlag = 0
doNN = 0
kNN = 20
dataFile = open('ml_new_train.csv','r')
testFile = open('ml_new_valid.csv','r')
done = 0
#numList = [0]
resFile = open('/home/x/ccrList.txt','w')
for numberTraining in numList:
	#print numberTraining
	done = 0
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

#print ftd[0][0]
	#print 'Computing TF-IDF!'
	tf, idf, tfidf = pp.getTfIdf( classTotal, textsWithWord, ftd, typeFlag )
	if doNN == 1:
		#getTfNN( N, nt, fnn, flagType )
		tfNN = pp.getTfNN( fnn )
	#print 'Predicting!'
	
	predClass, classList, CCR = pp.doTfIdfPrediction( testFile, tfidf, idf, isValidate)
	#predClass = pp.getTfidfNN( testFile, tfNN, idf, datClass, kNN, isValidate )
	#Ndoc = sum(classTotal)
	print numberTraining, CCR
	resFile.write(str(numberTraining) + ',' + str(CCR) + '\n')
resFile.close()
#	writeFile = open('/home/x/pred-corr.txt','w')
#	for ind,i in enumerate(predClass):
#		writeFile.write( str(classList[ind]) + ',' + str(predClass[ind]))
#		writeFile.write('\n')
