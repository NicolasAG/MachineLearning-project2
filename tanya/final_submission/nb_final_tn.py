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
import csv

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
# import sklearn
# from sklearn.ensemble import AdaBoostClassifier


def loaddata(dataset):
    #print 'Loading Dataset...', time.ctime()

    # load dataset
    df = pd.read_csv(dataset, sep=',', header=None, skiprows=2, low_memory=False)
    if dataset == 'ml_dataset_train.csv':
        df = df.drop(df.index[27313])

    return df


def nb():
    # Load Training Data, shuffle examples
    df = loaddata('ml_dataset_train.csv')
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

    # Load Testing Data
    # df = loaddata('ml_dataset_test_in.csv')
    # data = df[:].values.tolist()
    # xvalid = [el[1] for el in data]

    #print 'Parsing Dataset...', time.ctime()
    for i in range(0, len(x)):
        x[i] = x[i].lower()
        x[i] = x[i].replace('__eos__', "")
        x[i] = x[i].translate(string.maketrans("",""), string.punctuation)

    # non-useful words
    remove_list = ['and', 'you', 'my', 'on', 'got', 'who', 'what', 'where', 'when',
                   'why', 'for', 'that', 'like', 'let', 'by', 'im', 'not', 'to', 'but', 'be',
                   'if', 'did', 'ok', 'was', 'there', 'our', 'a', 'at', 'didnt', 'do', 'or', 'oh',
                   'have', 'youre', 'then', 'oh', 'of', 'in', 'is', 'hi', 'how', 'are', 'go', 'yeah',
                   'it', 'should', 'would', 'could', 'whats', 'yeah', 'its', 'so', 'an', 'it', '']

    # remove_list = [ '' ]

    # rl = stopwords.words('english')
    # remove_list = [rl[i].encode('ascii', 'ignore') for i in range(1, len(rl))]

    # remove non-useful words. wic = words in category
    wic_dict = [{}, {}, {}, {}]     # dictionary for each category
    master_dict = {}                # master dictionary
    print 'Building Classifier..', time.ctime()
    porter_stem = PorterStemmer()
    lancaster_stem = LancasterStemmer()
    snowball_stem = SnowballStemmer('english')
    wordnet_lem = WordNetLemmatizer()
    for i in range(0, len(x)):
        # split the string
        xsplit = str(x[i]).split(" ")
        x_seg = [word for word in xsplit if word not in remove_list]
        x_seg = [j.decode('utf-8') for j in x_seg]
        # Lemmatize!
        x_seg = [wordnet_lem.lemmatize(word) for word in x_seg]
        # x_seg = [porter_stem.stem(word) for word in x_seg]
        # x_seg = [lancaster_stem.stem(word) for word in x_seg]
        # x_seg = [snowball_stem.stem(word) for word in x_seg]
        x_seg = [j.encode('ascii', 'ignore') for j in x_seg]

        # for every word in the example:
        #   1) update our master list accordingly
        #   2) update the wic list accordingly

        # ADD THE WEIGHTS instead of ONE's
        for word in x_seg:
            if word in master_dict:
                master_dict[word] += 1
            else:
                master_dict[word] = 1

            if word in wic_dict[y[i]]:
                wic_dict[y[i]][word] += 1
            else:
                wic_dict[y[i]][word] = 1

    wsum = [0, 0, 0, 0]
    ysum = [0, 0, 0, 0]
    for i in range(0, 4):
        wsum[i] = sum(wic_dict[i].values())

    for ind in y:
        ysum[ind] += 1

    #################################################
    # now for the Naive Bayes...

    # P( class )
    p_class = [0, 0, 0, 0]
    sum_ysum = sum(ysum)
    for i in range(0, len(p_class)):
        p_class[i] = np.log(float(ysum[i]) / sum_ysum)

    print 'Calculating P(w | c)...', time.ctime()
    # P( word | class ) = number of occurances of word_i in class A / number of words in class A
    p_wgivenc = [{}, {}, {}, {}]
    for i in range(0, 4):
        wordsum = sum(wic_dict[i].values())
        for word in wic_dict[i]:
            p_wgivenc[i][word] = np.log(float(wic_dict[i][word]) / wordsum)

    # P( word ) = # occurances of word_i / total number of words
    p_word = master_dict
    nwords = sum(master_dict.values())
    for word, count in master_dict.items():
        p_word[word] = np.log(float(count)/nwords)

    print 'Classifying training set... ', time.ctime()
    numtrain = len(x)
    # create an array of results
    ytargettrain = []
    # for every example
    for i in range(0, numtrain):
        # split the string
        xsplit = str(x[i]).split(" ")
        x_seg = [word for word in xsplit if word not in remove_list]

        x_seg = [j.decode('utf-8') for j in x_seg]
        # Lemmatize!
        x_seg = [wordnet_lem.lemmatize(word) for word in x_seg]
        # x_seg = [porter_stem.stem(word) for word in x_seg]
        # x_seg = [lancaster_stem.stem(word) for word in x_seg]
        # x_seg = [snowball_stem.stem(word) for word in x_seg]
        x_seg = [j.encode('ascii', 'ignore') for j in x_seg]

        # initialize probabilities to the class probability
        p_classgivenword = [p_class[0], p_class[1], p_class[2], p_class[3]]

        # calc probability of it being in each of the classes
        for testclass in range(0, len(p_classgivenword)):
            # defprob = np.log(float(1) / (sum(wic_dict[testclass].values()) + 1))
            for word in x_seg:
                if word in p_wgivenc[testclass]:
                    p_classgivenword[testclass] += p_wgivenc[testclass][word] - p_word[word]
                # else:
                #     p_classgivenword[testclass] += defprob
        # take the maximum of the 4 values
        # print 'p_classgivenword', p_classgivenword
        ytargettrain.append(np.argmax(p_classgivenword))

    # #### TRAINING ERROR #####
    numcorrect = 0
    # ydiff = np.zeros((4, 4))
    for i in range(0, numtrain):
        # print 'ytargettrain, y', ytargettrain[i], y[i]
        if ytargettrain[i] == y[i]:
            numcorrect += 1
        # else:
        #     ydiff[ytargettrain[i], y[i]] += 1
    ccrtrain = float(numcorrect) / numtrain
    # print 'training error map:\n', ydiff
    print 'CCR Training: ', ccrtrain, '\n'

    # ###################################################
    # Check the validation
    numvalid = len(xvalid)
    print 'Parsing test set...', time.ctime()
    for i in range(0, numvalid):
        xvalid[i] = xvalid[i].lower()
        xvalid[i] = xvalid[i].replace('__eos__', "")
        xvalid[i] = xvalid[i].translate(string.maketrans("",""), string.punctuation)

    print 'Classifying test set... ', time.ctime()
    # create an array of results
    ytarget = []
    # for every example
    for i in range(0, numvalid):
        # initialize probabilities to the class probability
        p_classgivenword = [p_class[0], p_class[1], p_class[2], p_class[3]]
        xsplit = str(xvalid[i]).split(" ")
        x_seg = [word for word in xsplit if word not in remove_list]
        x_seg = [j.decode('utf-8') for j in x_seg]
        x_seg = [wordnet_lem.lemmatize(word) for word in x_seg]
        # x_seg = [porter_stem.stem(word) for word in x_seg]
        # x_seg = [lancaster_stem.stem(word) for word in x_seg]
        # x_seg = [snowball_stem.stem(word) for word in x_seg]
        x_seg = [j.encode('ascii', 'ignore') for j in x_seg]

        # calc probability of it being in each of the classes
        for testclass in range(0, len(p_classgivenword)):
            # defprob = np.log(float(1) / (sum(wic_dict[testclass].values()) + 1))

            for word in x_seg:
                if word in p_wgivenc[testclass]:
                    p_classgivenword[testclass] += p_wgivenc[testclass][word] - p_word[word]
                # else:
                #     p_classgivenword[testclass] += defprob

        # take the maximum of the 4 values
        ytarget.append(np.argmax(p_classgivenword))

    ##### VALIDATION ERROR #####
    numcorrect = 0
    ydiff = np.zeros((4, 4))
    for i in range(0, numvalid):
        #print 'guess, target', ytarget[i], yvalid[i]
        if ytarget[i] == yvalid[i]:
            numcorrect += 1
        else:
            ydiff[ytarget[i], yvalid[i]] += 1

    ccr = float(numcorrect) / numvalid
    print 'Finished...     ', time.ctime()

    # plot a histogram to see if our training and validation sets have similar distributions
    # plt.hist(y, bins=10, normed=True, alpha=0.5, color='b', label='Class Distribution - Training')
    # plt.hist(yvalid, bins=10, normed=True, alpha=0.5, color='r', label='Class Distribution - Validation (GT)')
    # plt.legend()
    # plt.show()

    # Training/Validation
    return ccr, ccrtrain, ydiff

    # Testing
    # return ytarget


def ccravg(numiter):
    ccrsum = 0
    ccrtrainsum = 0
    ydiffavg = np.zeros((4,4),dtype=float)
    print 'Iterating through naive bayes'
    for i in range(0, numiter):
        ccr, ccrtrain, ydiff = nb()
        ccrsum += ccr
        ccrtrainsum += ccrtrain
        ydiffavg += ydiff
        print 'iteration, ccr', i, ccr, '\n'

    ydiffavg = ydiffavg / numiter
    meanccr = ccrsum/numiter
    # print 'Test Error Map: \n', ydiffavg
    print 'CCR Average: ', meanccr
    print 'CCR Training Avg: ', ccrtrainsum/numiter

    plt.imshow(ydiffavg)
    plt.xlabel('Hypothesis'); plt.ylabel('Target')
    plt.colorbar(); plt.show()

    return meanccr


def print2csv(estimation):
    myfile = open('dataoutV2.csv', 'wb')
    wr = csv.writer(myfile)
    for value in estimation:
        wr.writerow(value)


def main(niter):

    # Training/Validation
    ccravg(niter)

    # Testing
    # ytarget = np.asmatrix(nb())
    # print2csv(ytarget.transpose())

main(1)

