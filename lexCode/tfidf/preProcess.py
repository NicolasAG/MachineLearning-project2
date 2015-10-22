#!/usr/bin/env python

# This file contains functions for doing pre-processing of data

import csv
import string
import nltk.corpus
import random
import numpy as np
import math
simpleWords = nltk.corpus.stopwords.words('english')
numFields = 3
numClasses = 4

# Four functions:
# getWordCount: 	Takes in data file, returns word counts for each class
# getProb:		Takes in word counts, spits out P(c) and P(w|c).
# doPrediction:		Takes in test file and trained probabilities, spits out list of predictions
# createTraining:	Takes in file and training count, spits out shuffled training/testing set.

# Input:
# dataFile:	String. csv file name
# doExclusion:	Int. Flag to exclude simple words
# dataLimit:	Int. Limit number of training examples
# 
# Output:
# classWordCount:	Dict list. Returns the number of times each word appears in a given class.
# classWordTotals:	Int list. Total number of words in each class
# classTotal:		Int list. Total number of examples in each class.
def getWordCount( dataFile, doExclusion, dataLimit ):
	csvReader = csv.reader(dataFile, delimiter=',', quotechar='"')
	indLine = 0
	datInd = []
	datClass = []
	wordList = []
	fnn = []
	classWordCount = [{} for i in range(numClasses)]
	classWordTotals= [0 for i in range(numClasses)]
	classTotal = [0 for i in range(numClasses)]
	textsWithWord = [{} for i in range(numClasses)]
	ftd = [[] for i in range(numClasses)]
#	pWC = [0 for k in range(numClasses)]

	for dirtyData in csvReader:
#		dirtyData = unicode(dirtyData,'utf-8')
		# First line - skip
		if indLine == 0:
			indLine += 1
			continue
#		if indLine % 500000 == 0:
#			print(indLine)
		if dataLimit != 0 and indLine >= dataLimit:
			return classWordCount, classWordTotals, classTotal
		if dirtyData == '':
			continue
#		print(indLine)
		# Get data index and class
		datInd.append( int(dirtyData[0].translate(None, string.punctuation)) )
		curClass = ( int(dirtyData[-1].translate(None,string.punctuation)) )
#		print(curClass)
		datClass.append(curClass)

		# dirtyData[1:-2] holds potentially large tokens.
		# The efficient way to do it would be to parse every token, then blah.
		# The I-don't-want-to-bother way is to smoosh them all together and parse from there.
		# Guess which one I'm doing!
		tempTok = ''
		for tok in dirtyData[1:-1]:
			tempTok += tok.replace('__EOS__','')
#		tempTok = unicode(tempTok,'utf-8')
		tempTok = tempTok.translate(None, string.punctuation).lower()
		tempList = string.split(tempTok)
		wordList = []
		for buf in tempList:
			wordList.append(unicode(buf,'utf-8'))
		
		# Count words	
		checkedWords = {}
		ftd[curClass].append({})
		fnn.append({})
		#print(ftd[curClass][-1])
		#print(ftd)
		for word in wordList:
			# If we're not removing simple words, or word is not a simple word
			if doExclusion == 0 or word not in simpleWords:
				classWordCount[curClass][word] = classWordCount[curClass].get(word, 0) + 1
				classWordTotals[curClass] += 1
#				print ftd
#				print ftd[curClass]
#				print ftd[curClass][-1]
				ftd[curClass][-1][word] = ftd[curClass][-1].get(word,0) + 1
				fnn[-1][word] = fnn[-1].get(word,0) + 1

				if checkedWords.get(word, -1) == -1:
					textsWithWord[curClass][word] = textsWithWord[curClass].get(word, 0) + 1
					checkedWords[word] = 1
			# Ignore simple words
			elif word == '':
				continue
			elif word in simpleWords:
				continue
#			if uniqueWords.get(word,0) == 0:
#				uniqueWords[word] = 1
#			if word not in uniqueWords:
#				uniqueWords.append(word)
		classTotal[curClass] += 1
		indLine += 1
		if dataLimit != 0 and indLine >= dataLimit:
			#print 'QUITTING'
			break
	return classWordCount, classWordTotals, classTotal, textsWithWord, ftd, fnn, datClass

# Calculates probabilities
def getProb( classWordCount, classWordTotals, classTotal ):
	# p( c )
	pc = [0 for k in range(numClasses)]
	pwc = [{} for k in range(numClasses)]
	for i in range(numClasses):
		pc[i] = float(classTotal[i]) / sum(classTotal)

	# P(wi | C)
	for classInd in range(numClasses):
		for word in classWordCount[classInd]:
			pwc[classInd][word] = float(classWordCount[classInd][word]) / classWordTotals[classInd]
	return pc, pwc

# Computes tf-idf for training data
def getTfIdf( N, nt, ftd, flagType ):
	# N: total number of documents in a class
	# nt: Number of docs in a class with particular term t
	# ftd: Frequency of words in examples
	# Start with tf = log(1+ftd)
	#tf = [[] for k in range(numClasses)]
	#tfnn = ftd
	tf = [{} for k in range(numClasses)]
	#idf = {}
	idf = [{} for k in range(numClasses)]
	#idf = [{} for k in range(numClasses)]
	classList = []
	for classInd in range(numClasses):
		#tf[classInd].append([])
		for docInd, doc in enumerate(ftd[classInd]):
			#tf[classInd][docInd].append({})
			for word in doc:
				# Add switch here for different ways of computing tf
				if flagType == 0:
					tf[classInd][word] = tf[classInd].get(word,0) + np.log(1+ftd[classInd][docInd].get(word, 0) )
				elif flagType == 1:
					tf[classInd][word] = tf[classInd].get(word,0) + np.log(1+ float(ftd[classInd][docInd].get(word, 0))/len(ftd[classInd][docInd]) )
					#idf[word] = np.log( sum(N) / (1.0 + float(nt[classInd][word]) ))
	
	# Tf now calculated
	# Do idf = log(1+N/nt)
	for classInd in range(numClasses):
		#idf[classInd].append({})
		for word in nt[classInd]:
			# Add switch here for different idf
			idf[classInd][word] = np.log(N[classInd]/(1.0+float(nt[classInd][word])))
			#idf[word] = np.log(sum(N)/(1.0+float(nt[classInd][word])))
	
	# Compute tf*idf
	tfIdf = [{} for k in range(numClasses)]
	for classInd in range(numClasses):
		for word in tf[classInd] and idf[classInd]:
			#print word
			tfIdf[classInd][word] = tf[classInd][word] * idf[classInd][word]

	return tf, idf, tfIdf

def getTfNN( fnn  ):
	#tf, idf, tfIdf = getTfIdf( N, nt, ftd, flagType )

	tfNN = []
	for docInd, doc in enumerate(fnn):
		tfNN.append({})
		for word in doc:
			tfNN[docInd][word] = np.log(1+fnn[docInd][word])
	return tfNN

# Does Naive Bayes prediction
def doPrediction( dataFile, pc, pwc, classWordTotal, isValidate ):
	csvReader = csv.reader(dataFile, delimiter=',', quotechar='"')
	defReturn = [1/float(tot) for tot in classWordTotal] #default return value for new words
	wordList = []
	classList = []
	good = 0 # number of correct classifications
	total = 0 # total number of examples
	for ind,dirtyData in enumerate(csvReader):
#		dirtyData = unicode(dirtyData,'utf-8')
		pcw = []
		# First line - skip
		if ind == 0:
			continue
#		if ind % 5000 == 0:
#			print(ind)
		if dirtyData == '':
			continue


		# Get data index and class
		if isValidate == 1:
			curClass = ( int(dirtyData[-1].translate(None,string.punctuation)) )
#			datClass.append(curClass)
		else:
			curClass = -1

		# The efficient way to do it would be to parse every token, then blah.
		# The I-don't-want-to-bother way is to smoosh them all together and parse from there.
		# Guess which one I'm doing!
		tempTok = ''
		for tok in dirtyData[1:-1]:
			tempTok += tok.replace('__EOS__','')
#		tempTok = unicode(tempTok,'utf-8')
		tempTok = tempTok.translate(None, string.punctuation).lower()
		tempList = string.split(tempTok)
		wordList = []
		for buf in tempList:
			wordList.append(unicode(buf,'utf-8'))

		# get P(C|w) = P(C) P(t|C) = P(C) prod(P(w|C)
		# Doing log(P(C|w)) instead
		# Initialize to P(C):
		# pTemp = [np.log(float(pc[i])) for i in range(numClasses)]

		# Calculate the probability of the set of words for each class
		for word in wordList:
			for indClass in range(numClasses):
				pTemp[indClass] += np.log( float(pwc[indClass].get(word, defReturn[indClass]) ) )

		# Get which class is most likely
		predClass = np.argmax( pTemp )
		classList.append(predClass)

		# If we can validation, track correct vs. total
		if isValidate == 1:
			if curClass == predClass:
				good += 1
			total += 1
	if isValidate == 0:
		return classList, 0
	else:
		return classList, float(good)/total

	
def doTfIdfPrediction( testFile, tfidf, idf, isValidate):
	csvReader = csv.reader(testFile, delimiter=',', quotechar='"')
#	defReturn = [1/float(tot) for tot in classWordTotal] #default return value for new words
	wordList = []
	classList = []
	predClass = []
	good = 0 # number of correct classifications
	total = 0 # total number of examples
	trainMag = [0 for k in range(numClasses)]
	CCR = 0
	for classInd in range(numClasses):
		for word in tfidf[classInd]:
			trainMag[classInd] += tfidf[classInd][word]**2
		trainMag[classInd] = trainMag[classInd] ** (0.5)

	for ind,dirtyData in enumerate(csvReader):
#		dirtyData = unicode(dirtyData,'utf-8')
		tf = {}
		# First line - skip
		if ind == 0:
			continue
#		if ind % 5e6 == 0:
#			print(ind)
		if dirtyData == '':
			continue


		# Get data index and class
		if isValidate == 1:
			curClass = ( int(dirtyData[-1].translate(None,string.punctuation)) )
#			datClass.append(curClass)
		else:
			curClass = -1

		# The efficient way to do it would be to parse every token, then blah.
		# The I-don't-want-to-bother way is to smoosh them all together and parse from there.
		# Guess which one I'm doing!
		tempTok = ''
		for tok in dirtyData[1:-1]:
			tempTok += tok.replace('__EOS__','')
#		tempTok = unicode(tempTok,'utf-8')
		tempTok = tempTok.translate(None, string.punctuation).lower()
		tempList = string.split(tempTok)
		wordList = []
		for buf in tempList:
			wordList.append(unicode(buf,'utf-8'))


		# Get TF of wordList
		for word in wordList:
			tf[word] = tf.get(word, 0) + 1
		for word in tf:
			tf[word] = np.log(1+tf[word])
		
		dotVec = [0 for k in range(numClasses)]
		buf = [0 for k in range(numClasses)]
		distClass = [0 for k in range(numClasses)]
		for classInd in range(numClasses):
			qMag = 0
			bufMag = 0
			for word in tf:
				dotVec[classInd] += tf[word] * idf[classInd].get(word, 0) * tfidf[classInd].get(word,0)
				#buf[classInd] += tf[word] * idf[classInd].get(word,0) - tfidf[classInd].get(word,0)
				#qMag += (tf[word] * idf.get(word,0)) ** 2
				qMag += (tf[word] * idf[classInd].get(word,0)) ** 2
				bufMag += buf[classInd]**2
			qMag = qMag ** (0.5)
			qMag = np.max([1e-10, qMag])
			dotVec[classInd] = np.abs( np.arccos( dotVec[classInd] / qMag / trainMag[classInd] ))

			#distClass[classInd] = bufMag ** 0.5
			#dotVec[classInd] = np.arccos( dotVec[classInd] / qMag  )
		predClass.append( np.argmin( dotVec ) )
		#predClass.append( np.argmin(distClass))
		# Get IDF of wordList, assuming different class
		#sumTfidf = [0 for i in range(numClasses)]
		#for classInd in range(numClasses):
		#	for word in wordList:
		#		sumTfidf[classInd] += tf[word] * idf[classInd].get(word, 0)
		#predClass.append( np.argmax( sumTfidf))


		if isValidate == 1:
			if predClass[-1] == curClass:
				good += 1
			classList.append(curClass)
			total += 1
			#print float(good)/total
#	print total
	CCR = float(good)/total
	#print CCR
	return predClass, classList, CCR
#classTotal, textsWithWord, fnn, datClass typeFlag

def getTfidfNN( testFile, tfnn, idf, datClass, kNN, isValidate ):
	csvReader = csv.reader(testFile, delimiter=',', quotechar='"')
#	defReturn = [1/float(tot) for tot in classWordTotal] #default return value for new words
	wordList = []
	classList = []
	predClass = []
	good = 0 # number of correct classifications
	total = 0 # total number of examples
	trainMag = [0 for k in range(numClasses)]
	CCR = 0
	for classInd in range(numClasses):
		for word in tfnn[classInd]:
			trainMag[classInd] += tfnn[classInd][word]**2
		trainMag[classInd] = trainMag[classInd] ** (0.5)

	for ind,dirtyData in enumerate(csvReader):
#		dirtyData = unicode(dirtyData,'utf-8')
		tf = {}
		# First line - skip
		if ind == 0:
			continue
		if ind % 500 == 0:
			print(ind)
		if dirtyData == '':
			continue


		# Get data index and class
		if isValidate == 1:
			curClass = ( int(dirtyData[-1].translate(None,string.punctuation)) )
#			datClass.append(curClass)
		else:
			curClass = -1

		# The efficient way to do it would be to parse every token, then blah.
		# The I-don't-want-to-bother way is to smoosh them all together and parse from there.
		# Guess which one I'm doing!
		tempTok = ''
		for tok in dirtyData[1:-1]:
			tempTok += tok.replace('__EOS__','')
#		tempTok = unicode(tempTok,'utf-8')
		tempTok = tempTok.translate(None, string.punctuation).lower()
		tempList = string.split(tempTok)
		wordList = []
		for buf in tempList:
			wordList.append(unicode(buf,'utf-8'))
		
		# Get TF of test sample
		# First get frequency
		tf = {}
		for word in wordList:
			tf[word] = tf.get(word, 0) + 1
		# Get TF
		tfidf = [{} for k in range(numClasses)]
		for word in tf:
			tf[word] = np.log(1+tf.get(word,0) )
			for classInd in range(numClasses):
				tfidf[classInd][word] = tf[word] * idf[classInd].get(word,0)
		tMag = [0 for k in range(numClasses)]
		
		for word in tf:
			for classInd in range(numClasses):
				tMag[classInd] += (tfidf[classInd][word]) ** 2
		for classInd in range(numClasses):
			tMag[classInd] = tMag[classInd] ** 0.5
		print tMag
		dotVec = []#[] for k in range(numClasses)]
		# Get similarity
		dMag = []
		for docInd, doc in enumerate(tfnn):
			dotVec.append([])
			#dMag = [0 for k in range(numClasses)]
			#dotVec[-1][datClass[docInd]] = 0
			curDatClass = datClass[docInd]
			#dotVec[-1][curDatClass] = 0
			dotVec[-1] = 0
			dMag = 0
			for word in doc:
			#for word in tfidf[datClass[docInd]]:
				buf = doc.get(word, 1e-10) * idf[curDatClass].get(word,0)
				dotVec[-1] += tfidf[curDatClass].get(word,0) * buf
				dMag += buf ** 2
			
			dMag = dMag ** 0.5
			preOp = dotVec[-1]
			dotVec[-1] = np.abs( np.arccos( dotVec[-1] / tMag[curDatClass] / dMag ))
			if math.isnan(dotVec[-1]):
				print 'Start:'
				#print preOp
				print tMag[curDatClass]
				#print dMag
			#if math.isnan(dotVec[-1]):
				#print( 'dMag:' )
				#print( dMag )
				#print(' tMag:' )
				#print( tMag[curDatClass] )
				#print dotVec[-1]/tMag[curDatClass]/dMag
			
			#print(dotVec[-1] / tMag[curDatClass] / dMag)

#			for classInd in range(numClasses):
#				dotVec[-1][classInd] = 0
#				dMag = 0
#				for word in tfidf[classInd]:
#					buf = doc.get(word,1e-10) * idf[datClass(docInd)].get(word,1e-10)
#					dotVec[-1][classInd] += tfidf[classInd][word] * buf
#					dMag += buf ** 2
#				dMag = dMag ** 0.5
#				dotVec[-1][classInd] = np.abs( np.arccos( dotVec[-1][classInd] / tMag[classInd] / dMag ))

		# Get K NN
		nnList = [0 for k in range(numClasses)]
		dotList = np.argsort(dotVec )
#		print(type(dotList))
#		print(len(dotList))
#		for k in range(2*kNN):
#			print dotVec[ dotList[k]]
		# Get K last classes
		for k in range(kNN):
#			print dotList[-k-1]
			ind = datClass[ dotList[-k-1] ]
#			print(ind)
			nnList[ ind ] += 1
		predClass.append( np.argmax(nnList) )
#		print( predClass )
#		print( curClass )
		print( predClass[-1], curClass )
		if predClass[-1] == curClass:
			good += 1
		total += 1
		print float(good)/total

	return predClass







def readData( dataFile, isTest ):
	csvReader = csv.reader(dataFile, delimiter=',', quotechar='"')
	wordList = []
	dataIndex = []
	classIndex = []
	for dirtyData in csvReader:
#		dirtyData = unicode(dirtyData,'utf-8')
		# First line - skip
		if indLine == 0:
			indLine += 1
			continue
		if indLine % 5000 == 0:
			print(indLine)
		if dirtyData == '':
			continue
#		print(indLine)
		# Get data index and class
		datInd.append( int(dirtyData[0].translate(None, string.punctuation)) )
		curClass = ( int(dirtyData[-1].translate(None,string.punctuation)) )
#		print(curClass)
		datClass.append(curClass)

		# dirtyData[1:-2] holds potentially large tokens.
		# The efficient way to do it would be to parse every token, then blah.
		# The I-don't-want-to-bother way is to smoosh them all together and parse from there.
		# Guess which one I'm doing!
		tempTok = ''
		for tok in dirtyData[1:-1]:
			tempTok += tok.replace('__EOS__','')
#		tempTok = unicode(tempTok,'utf-8')
		tempTok = tempTok.translate(None, string.punctuation).lower()
		tempList = string.split(tempTok)
		wordList = []
		for buf in tempList:
			wordList.append(unicode(buf,'utf-8'))

	return wordList


def createTraining( fileIn, fileTrainOut, fileValidOut, numTraining):
	dataIn = open(fileIn, 'r')
	dataTrainOut = open(fileTrainOut, 'w')
	dataValidOut = open(fileValidOut, 'w')
	random.seed('chicken')
	datAll = []
	for ind,lin in enumerate(dataIn):
		if ind == 0:
			continue
		else:
			datAll.append(lin)

	random.shuffle(datAll)
#	print datAll
	dataTrainOut.write('Id,Interview,Prediction\n')
	dataValidOut.write('Id,Interview,Prediction\n')
	for ind,dat in enumerate(datAll):
		if ind < numTraining:
			dataTrainOut.write(dat)
		else:
			dataValidOut.write(dat)
	dataIn.close()
	dataTrainOut.close()
	dataValidOut.close()
	

def checkTfIdf(tf, idf, tfidf, word):
	for i in range(numClasses):
		print tf[i].get(word,0)
		print idf[i].get(word,0)
		print tfidf[i].get(word,0)
		print ''