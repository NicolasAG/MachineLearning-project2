# ***************************************************************************
# Naive Bayes Implementation With Adaboost
# Purpose: Accurately predict category of interview text (0-3)
#           from the given training dataset. Apply AdaBoost algorithm
# Author: Tanya Nair

import time
import string
import csv
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer


class NaiveBayes:
    def __init__(self, startflag, boost=False, partition=True):
        #print 'Initializing...'
        self.startflag = startflag
        self.boost = boost
        self.partition = partition
        if startflag:
            self.data = []
        self.xall = []
        self.yall = []
        self.remove_list = ['and', 'you', 'my', 'on', 'got', 'who', 'what', 'where', 'when',
                            'why', 'for', 'that', 'like', 'let', 'by', 'im', 'not', 'to', 'but', 'be',
                            'if', 'did', 'ok', 'was', 'there', 'our', 'a', 'at', 'didnt', 'do', 'or', 'oh',
                            'have', 'youre', 'then', 'oh', 'of', 'in', 'is', 'hi', 'how', 'are', 'go', 'yeah',
                            'it', 'should', 'would', 'could', 'whats', 'yeah', 'its', 'so', 'an', 'it', '']
        # remove_list = [ '' ]
        # remove_list = (stopwords.words('english')).encode('ascii', 'ignore')

        self.wic_dict = [{}, {}, {}, {}]     # dictionary for each category
        self.master_dict = {}                # master dictionary
        # self.porter_stem = PorterStemmer()
        # self.lancaster_stem = LancasterStemmer()
        # self.snowball_stem = SnowballStemmer('english')
        self.wordnet_lem = WordNetLemmatizer()
        self.wsum = [0, 0, 0, 0]
        self.ysum = [0, 0, 0, 0]
        self.p_class = [0, 0, 0, 0]
        self.p_wgivenc = [{}, {}, {}, {}]
        self.p_word = []
        self.ytarget = []
        # for training
        self.ytargettrain = []

        # stuff for AdaBoost
        if self.startflag and boost:
            self.nexamples = 1 #len(self.x)
            self.weights = [] #np.ones(self.nexamples)/self.nexamples # initial value
            # self.HYP = []
            self.ALPHA = []
            self.pwgc_master = []

    def loadtestdata(self, myfile):
        df = pd.read_csv(myfile, sep=',', header=None, skiprows=2, low_memory=False)
        data = df[:].values.tolist()
        return data

    def loadtraindata(self, myfile):
        if self.startflag:
            'loading data'
            self.startflag = False
            df = pd.read_csv(myfile, sep=',', header=None, skiprows=2, low_memory=False)
            df = df.drop(df.index[27313])
            data = df[:].values.tolist()
            random.shuffle(data)
            self.data = data
            self.xall = [el[1] for el in data]
            self.yall = [el[2] for el in data]

        else:
            self.xall = [el[1] for el in self.data]
            self.yall = [el[2] for el in self.data]

        if self.partition:
            tvpart = 40000
            self.x = self.xall[0:tvpart]
            self.xtest = self.xall[tvpart:]
            self.y = self.yall[0:tvpart]
            self.ytest = self.yall[tvpart:]
        else:
            print 'Target: Unclassified Test Set'
            self.x = self.xall
            self.y = self.yall
            self.testdata = self.loadtestdata('ml_dataset_test_in.csv')
            self.xtest = [el[1] for el in self.testdata]

        if self.boost:
            # update nexamples and weights now that we've input the data
            self.nexamples = len(self.x)
            self.weights = np.ones(self.nexamples)/self.nexamples

    def parsedata(self):
        for i in range(len(self.x)):
            self.x[i] = self.x[i].lower()
            self.x[i] = self.x[i].replace('__eos__', "")
            self.x[i] = self.x[i].translate(string.maketrans("", ""), string.punctuation)

        for i in range(len(self.xtest)):
            self.xtest[i] = self.xtest[i].lower()
            self.xtest[i] = self.xtest[i].replace('__eos__', "")
            self.xtest[i] = self.xtest[i].translate(string.maketrans("", ""), string.punctuation)

    def countwords(self):
        if self.boost:
            scaleval = self.weights
        else:
            scaleval = 1

        for i in range(0, len(self.x)):

            # split the string
            xsplit = str(self.x[i]).split(" ")
            x_seg = [word for word in xsplit if word not in self.remove_list]
            x_seg = [j.decode('utf-8') for j in x_seg]
            # Lemmatize!
            x_seg = [self.wordnet_lem.lemmatize(word) for word in x_seg]
            x_seg = [j.encode('ascii', 'ignore') for j in x_seg]

            # for every word in the example:
            #   1) update our master list accordingly
            #   2) update the wic list accordingly
            # ADD THE WEIGHTS instead of ONE's
            for word in x_seg:
                if word in self.master_dict:
                    self.master_dict[word] += 1 * scaleval[i]
                else:
                    self.master_dict[word] = 1 * scaleval[i]

                if word in self.wic_dict[self.y[i]]:
                    self.wic_dict[self.y[i]][word] += 1 * scaleval[i]
                else:
                    self.wic_dict[self.y[i]][word] = 1 * scaleval[i]

        # print self.wic_dict
        for i in range(0, 4):
            self.wsum[i] = sum(self.wic_dict[i].values())

        for ind in self.y:
            self.ysum[ind] += 1

    def learn(self):
        print 'Training...', time.ctime()
        self.countwords()
        # P( class )
        sum_ysum = sum(self.ysum)
        for i in range(0, len(self.p_class)):
            self.p_class[i] = np.log(float(self.ysum[i]) / sum_ysum)

        # P( word | class ) = number of occurrences of word_i in class A / number of words in class A
        for i in range(0, 4):
            wordsum = sum(self.wic_dict[i].values())
            for word in self.wic_dict[i]:
                self.p_wgivenc[i][word] = np.log(float(self.wic_dict[i][word]) / wordsum)

        self.p_word = self.master_dict
        # P( word ) = # occurances of word_i / total number of words
        nwords = sum(self.master_dict.values())
        for word, count in self.master_dict.items():
            self.p_word[word] = np.log(float(count)/nwords)

    def trainerr(self):
        print 'Classifying training set... ', time.ctime()
        numtrain = len(self.x)
        # for every example
        for i in range(0, numtrain):
            # split the string
            xsplit = str(self.x[i]).split(" ")
            x_seg = [word for word in xsplit if word not in self.remove_list]
            x_seg = [j.decode('utf-8') for j in x_seg]
            # Lemmatize!
            x_seg = [self.wordnet_lem.lemmatize(word) for word in x_seg]
            x_seg = [j.encode('ascii', 'ignore') for j in x_seg]

            # initialize probabilities to the class probability
            p_classgivenword = [self.p_class[0], self.p_class[1], self.p_class[2], self.p_class[3]]
            # calc probability of it being in each of the classes
            for testclass in range(0, len(p_classgivenword)):
                for word in x_seg:
                    if word in self.p_wgivenc[testclass]:
                        p_classgivenword[testclass] += self.p_wgivenc[testclass][word] - self.p_word[word]

            self.ytargettrain.append(np.argmax(p_classgivenword))

        # #### TRAINING ERROR #####
        numcorrect = 0
        ydiff = np.zeros((4, 4))
        for i in range(0, numtrain):
            # print 'ytargettrain, y', self.ytargettrain[i], self.y[i]
            if self.ytargettrain[i] == self.y[i]:
                numcorrect += 1
            else:
                ydiff[self.ytargettrain[i], self.y[i]] += 1
        ccrtrain = float(numcorrect) / numtrain
        #print 'training error map:\n', ydiff
        print 'CCR Training: ', ccrtrain

    def validerr(self):
        numcorrect = 0
        numvalid = len(self.xtest)
        ydiff = np.zeros((4, 4))
        for i in range(0, numvalid):
            #print 'guess, target', ytarget[i], yvalid[i]
            if self.ytarget[i] == self.ytest[i]:
                numcorrect += 1
            else:
                ydiff[self.ytarget[i], self.ytest[i]] += 1

        # return the CCR
        return float(numcorrect) / numvalid

    def classifytest(self):
        numvalid = len(self.xtest)
        # numvalid = 20
        print 'Classifying... ', time.ctime()
        # for every example
        for i in range(numvalid):
            xsplit = str(self.xtest[i]).split(" ")
            x_seg = [word for word in xsplit if word not in self.remove_list]
            x_seg = [j.decode('utf-8') for j in x_seg]
            x_seg = [self.wordnet_lem.lemmatize(word) for word in x_seg]
            x_seg = [j.encode('ascii', 'ignore') for j in x_seg]

            p_cgw = []
            for j in range(len(self.ALPHA)):
                # initialize probabilities to the class probability
                p_classgivenword = [self.p_class[0], self.p_class[1], self.p_class[2], self.p_class[3]]
                p_wgivenc = self.pwgc_master[j]
                # calc probability of it being in each of the classes
                for testclass in range(len(p_classgivenword)):
                    for word in x_seg:
                        if word in self.p_wgivenc[testclass]:
                            p_classgivenword[testclass] += p_wgivenc[testclass][word] - self.p_word[word]

                p_cgw.append(np.multiply(self.ALPHA[j], p_classgivenword))

            # take the maximum of the 4 values
            # self.ytarget.append(np.argmax(p_classgivenword))
            self.ytarget.append(np.argmax(np.max(p_cgw, axis=0)))

    def plotresults(self):
        # plot a histogram to see if our training and validation sets have similar distributions
        plt.hist(self.y, bins=10, normed=True, alpha=0.5, color='b', label='Class Distribution - Training')
        plt.hist(self.ytest, bins=10, normed=True, alpha=0.5, color='r', label='Class Distribution - Validation (GT)')
        plt.legend()
        plt.show()

    def plotydiffavg(self, ydiffavg):
        plt.imshow(ydiffavg)
        plt.xlabel('Classification')
        plt.ylabel('Ground Truth')
        plt.colorbar()
        plt.show()

    def print2csv(self):
        myfile = open('dataout.csv', 'wb')
        wr = csv.writer(myfile)
        hypothesis = np.asmatrix(self.ytarget).transpose()
        for value in hypothesis:
            wr.writerow(value)

    def run_nb(self, boost, startflag, partition):
        self.__init__(boost=boost, startflag=startflag, partition=partition)
        self.loadtraindata('ml_dataset_train.csv')
        self.parsedata()
        self.learn()
        self.trainerr()
        # self.classifytest()
        # return self.validerr() # validerr returns the CCR

    def ccravg(self, numiter, boost=False):
        ccrsum = 0
        ydiffavg = np.zeros((4, 4), dtype=float)
        print "Iterating through Naive Bayes... ", time.ctime()
        for i in range(numiter):
            self.__init__(boost=boost)
            ccr = self.run_nb()
            ccrsum += ccr
            ydiffavg += ydiffavg
            print "Iteration, CCR", i, ccr, '\n'

        ydiffavg = ydiffavg / numiter
        meanccr = ccrsum / numiter

        #self.plotydiffavg(ydiffavg)

        return meanccr

    def run_nb_testing(self):
        self.__init__(boost=False, partition=False)
        self.loadtraindata('ml_dataset_train.csv')
        self.parsedata()
        self.learn()
        self.classifytest()
        self.print2csv()

    def boosterrun(self, niter, test=False):
        for i in range(niter):
            print 'BOOST ITERATION:   ', i
            # train the weak learner
            if i == 0:
                self.run_nb(boost=True, startflag=True, partition=True)
            else:
                self.run_nb(boost=True, startflag=False, partition=True)

            self.pwgc_master.append(self.p_wgivenc)
            hypothesis = self.ytargettrain

            # make an array of all the examples we get wrong
            errors = np.array([self.y[i] != hypothesis[i] for i in range(len(hypothesis))])
            # self.err_master.append(errors)
            err = (errors*self.weights).sum()

            if test:
                return err

            alpha = np.log(float(1-err) / err) + np.log(4)
            print 'err, alpha: ', err, alpha, '\n'

            w = np.zeros(self.nexamples)
            # update the weight for each example: misclassification --> increased weight
            for j in range(self.nexamples):
                if errors[j] == 1:
                    w[j] = self.weights[j] * np.exp(alpha)
                else:
                    w[j] = self.weights[j] * np.exp(-alpha)

            # update self.weights with normalization
            self.weights = w / w.sum()
            # self.HYP.append(hypothesis)
            self.ALPHA.append(alpha)

    def adaboost(self, niter=5):
        # booster runs 5x by default
        self.boosterrun(niter)
        self.classifytest()
        ccr = self.validerr()
        print 'ccr on test set: ', ccr

    def ttest(self, count):
        while( count > 0):
            count -= 1
            print '************'
            print "MASTER LOOP: ", count
            self.adaboost(100)

if __name__ == '__main__':
    print 'Start'
    q = NaiveBayes(True)
    q.adaboost(20)

