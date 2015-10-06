#!/usr/bin/env python

import csv, random

"""
Read the data from a CSV file and put it all in an array.
Assume a title line at the begining of the file.
@param path - the path to the file to read.
@param isTraining - flag to tell if we are accessing the training data data or not. (default is True).
@param returnSize - the number of examples to return. (default is -1 which means ALL data).
@return - an array of tuples of the form (id, text, class number).
example of data: [
    (id1, text1, class#),
    (id2, text2, class#),
    (id3, text3, class#)
]
"""
def readData(path, isTraining=True, returnSize=-1):
    data = []
    with open(path, 'rb') as datafile:
        reader = csv.reader(datafile)
        line = 0
        for row in reader: # go through each line:
            line = line+1
            if line == 1:  # skip the 1st line (title line)
                continue
            if len(row) < 2: # skip empty lines
                continue

            lID = row[0]
            lText = row[1]
            lClass = None

            if isTraining:
                lClass = row[2]
            
            data.append((lID, lText, lClass)) # add the tuple for this example.

    random.shuffle(data) # randomize the examples
    
    if returnSize > 0:
    	return random.sample(data, returnSize) # for testing purposes return a small sample.
    else:
    	return data


"""
Create a csv file with given data
@param data - a 2D array with row = example
"""
def writeData(data):
    with open("./data/ml_dataset_test_out.csv", 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Id", "Prediction"])
        for row in data:
            writer.writerow(row)


"""
Uncomment to test:
"""
#path_to_train_data = "./data/ml_dataset_train.csv"
#path_to_test_data = "./data/ml_dataset_test_in.csv"
#training_data = readData(path_to_train_data, returnSize=10)
#print training_data
