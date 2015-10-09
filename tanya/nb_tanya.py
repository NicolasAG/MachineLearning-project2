# ***************************************************************************
# Naive Bayes Implementation
# Purpose: Accurately predict category of interview text (0-3)
#           from the given training dataset
# Author: Tanya Nair

import numpy as np
import pandas as pd
import time
import string
import random
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

def loaddata():
    print 'Loading Dataset...', time.ctime()

    # load dataset
    df = pd.read_csv('ml_dataset_train.csv', sep=',', header=None, skiprows=2, low_memory=False)
    df = df.drop(df.index[27313])
    return df


def nb():
    # shuffle examples
    df = loaddata()
    data = df[:].values.tolist()
    random.shuffle(data)
    xall = [el[1] for el in data]
    yall = [el[2] for el in data]

    # train/validation partition
    tvpart = 40000
    x = xall[0:tvpart]
    xvalid = xall[tvpart:]
    y = yall[0:tvpart]
    yvalid = yall[tvpart:]

    print 'Parsing Dataset...', time.ctime()
    for i in range(0, len(x)):
        x[i] = x[i].lower()
        x[i] = x[i].replace('__eos__', "")
        x[i] = x[i].translate(string.maketrans("",""), string.punctuation)

    # non-useful words
    remove_list = ['and', 'you', 'my', 'on', 'got', 'who', 'what', 'where', 'when',
                   'why', 'for', 'that', 'like', 'let', 'by', 'im', 'not', 'to', 'but', 'be',
                   'if', 'did', 'ok', 'was', 'there', 'our', 'a', 'at', 'didnt', 'do', 'or', 'oh'
                   'have', 'youre', 'then', 'oh', 'of', 'in', 'is', 'hi', 'how', 'are', 'go', 'yeah'
                   'it', 'should', 'would', 'could', 'whats', 'yeah', 'its', 'so', 'an', 'it', '']

    #remove_list = [ '' ]

    #remove_list = (stopwords.words('english')).encode('ascii', 'ignore')

    # remove non-useful words. wic = words in category
    wic_dict = [{}, {}, {}, {}]     #dictionary for each category
    print 'Building Classifier..', time.ctime()
    for i in range(0, len(x)):
        # split the string
        xsplit = str(x[i]).split(" ")
        x_seg = [word for word in xsplit if word not in remove_list]
        #print x_seg

        # for every word in the example:
        #   1) update our master list accordingly
        #   2) update the wic list accordingly
        for word in x_seg:
            # if word in master_dict:
            #     master_dict[word] += 1
            # else:
            #     master_dict[word] = 1
            if word in wic_dict[y[i]]:
                wic_dict[y[i]][word] += 1
            else:
                wic_dict[y[i]][word] = 1

    wsum=[0,0,0,0]
    ysum=[0,0,0,0]
    for i in range(0,4):
        wsum[i] = sum(wic_dict[i].values())

    for ind in y:
        ysum[ind] += 1

    #################################################
    # now for the Naive Bayes...

    # P( class )
    p_class = [0,0,0,0]
    sum_ysum = sum(ysum)
    for i in range(0, len(p_class)):
        p_class[i] = np.log(float(ysum[i]) / sum_ysum)

    #print p_class

    print 'Calculating P(w | c)...', time.ctime()
    # P( word | class ) = number of occurances of word_i in class A / number of words in class A
    p_wgivenc = [{}, {}, {}, {}]
    for i in range (0,4):
        wordsum = sum(wic_dict[i].values())
        for word in wic_dict[i]:
            p_wgivenc[i][word] = np.log(float(wic_dict[i][word]) / wordsum)

    # ###################################################
    # Check the validation
    numvalid = len(xvalid)
    print 'Parsing validation set...', time.ctime()
    for i in range(0, numvalid):
        xvalid[i] = xvalid[i].lower()
        xvalid[i] = xvalid[i].replace('__eos__', "")
        xvalid[i] = xvalid[i].translate(string.maketrans("",""), string.punctuation)

    print 'Classifying validation set... ', time.ctime()
    # now for every validation case, do this for each class
    #           1) for word_i in xvalid_string: product( p_wgivenc[i][word] )
    #           2) multiply this by p_class[i]
    # then compare the 4 values, take the biggest one

    # create an array of results
    ytarget = []
    # for every example
    for i in range(0, numvalid):
        # initialize probabilities to the class probability
        p_classgivenword = [p_class[0], p_class[1], p_class[2], p_class[3]]
        xsplit = str(xvalid[i]).split(" ")
        x_seg = [word for word in xsplit if word not in remove_list]

        # calc probability of it being in each of the classes
        for testclass in range(0,len(p_classgivenword)):

            defprob = np.log(float(1) /(sum(wic_dict[testclass].values()) +1))
            for word in x_seg:
                #p_wgivenc[testclass].get(word, DEFVAL)

                if word in p_wgivenc[testclass]:
                    p_classgivenword[testclass] += p_wgivenc[testclass][word]
                else:
                    #print 'word not in training set: ', word
                    p_classgivenword[testclass] += defprob

            # sanity check - are any of the probs 0's?
            #print 'prob', p_classgivenword[testclass]

        # take the maximum of the 4 values
        ytarget.append(np.argmax(p_classgivenword))

    # compare the calculated target with the result
    numcorrect = 0
    ydiff = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    for i in range (0,numvalid):
        #print 'guess, target', ytarget[i], yvalid[i]
        if ytarget[i] == yvalid[i]:
            numcorrect += 1
        else:
            ydiff[ytarget[i],yvalid[i]] += 1



    ccr = float(numcorrect) / numvalid
    #print 'numcorrect, cases, CCR:      ', numcorrect, numvalid, ccr
    print 'Finished...     ', time.ctime()

    # plot a histogram to see if our training and validation sets have similar distributions
    plt.hist(y, bins=10, normed=True, alpha=0.5, color='b', label='Class Distribution - Training')
    plt.hist(yvalid, bins=10, normed=True, alpha=0.5, color='r', label='Class Distribution - Validation (GT)')
    plt.legend()
    plt.show()


    return ccr, ydiff

def ccravg(numiter):
    ccrsum = 0
    ydiffavg = np.zeros((4,4),dtype=float)
    print 'Iterating through naive bayes'
    for i in range(0, numiter):
        ccr, ydiff = nb()
        ccrsum += ccr
        ydiffavg += ydiff
        print 'iteration, ccr', i, ccr, '\n'

    ydiffavg = ydiffavg / numiter
    meanccr = ccrsum/numiter
    plt.imshow(ydiffavg)
    plt.xlabel('Classification'); plt.ylabel('Ground Truth')
    plt.colorbar(); plt.show()
    print ydiffavg
    return meanccr


def main(niter):

    meanccr = ccravg(niter)
    print '\n\naverage: ', meanccr

main(5)

