import pandas as pd
import numpy as np
from FCA import FCA

Accuracy = 0
TP = 0
TN = 0
FP = 0
FN = 0

train          = pd.read_csv('data/train.csv')
train_positive = train[train['common_flares'] == 1]
train_negative = train[train['common_flares'] == 2]

train_positive = np.array(train_positive)
train_negative = np.array(train_negative)

test          = pd.read_csv('data/test.csv')
test_positive = test[test['common_flares'] == 1]
test_negative = test[test['common_flares'] == 2]

######################################################################################

def f(test, train_positive, train_negative, vp, vn):
    guess = []

    my_test = np.array(test)
    for i in my_test:
        cf = i[5]
        i = np.array([i])
        guess.append([fca.fca(i, train_positive, train_negative, vp, vn), cf])

    for index in range(len(guess)):
        if guess[index][1] == 1:
            if guess[index][0][0] >= 50:
                guess[index].append(1)
            else:
                guess[index].append(0)

        if guess[index][1] == 2:
            if guess[index][0][1] >= 50:
                guess[index].append(2)
            else:
                guess[index].append(0)

    return guess

######################################################################################

fca = FCA(0)

guess = f(test, train_positive, train_negative, 0.95, 1)

for v in np.array(guess):
    if v[2] > 0:
        Accuracy += 1
        if v[1] == 1:
            TP +=1
        if v[1] == 2:
            TN +=1

Accuracy = Accuracy / test.shape[0]
FP = test_positive.shape[0] - TP
FN = test_negative.shape[0] - TN

precision = (TP) / (TP + FP)
recall    = (TP) / (TP+FN)
F         = 2 * ((precision * recall) / (precision + recall))

print(f"TP is: {TP}, FP is : {FP}")
print(f"TN is: {TN}, FN is : {FN}")
print(f"Accuracy is :{Accuracy}")
print(f"Precision is :{precision}")
print(f"Recall is :{recall}")
print(f"F is :{F}")