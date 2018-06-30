import pyautogui
import cv2

import numpy as np
import pandas as pd
import time
import math
import sys
import csv
import PIL

import parse_game
from sklearn import tree, svm, datasets, linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
##
link = 'http://www.freeminesweeper.org//minecore.html'

box = (0,0,550,450)
"""
0 - img
1 - coordinates
2 - number
3- label
"""
pics = {'reset':[cv2.imread(".\\pics\\reset.jpg", 0), [], 1000, 'reset'],
    'nan': [cv2.imread('.\\pics\\blank.jpg',0), [], -1, 'blank'],
    'zero':[cv2.imread('.\\pics\\open0.jpg',0), [], 0, '0'],
    'one':[cv2.imread('.\\pics\\open1.jpg',0), [], 1 , '1'],
    'two':[cv2.imread('.\\pics\\open2.jpg',0), [], 2, '2'],
    'three':[cv2.imread('.\\pics\\open3.jpg',0), [], 3, '3'],
    'four':[cv2.imread('.\\pics\\open4.jpg',0), [], 4, '4'],
    'bomb':[cv2.imread('.\\pics\\bombrevealed.jpg',0),[], -100, 'bomb'],
    'bomb':[cv2.imread('.\\pics\\bombflagged.jpg',0),[], -10, 'bombFlagged']
}


resetImg = cv2.imread(".\\pics\\reset.jpg", 0)
nanImg = cv2.imread('.\\pics\\-1.jpg',0)
zeroImg = cv2.imread('.\\pics\\0.jpg',0)
oneImg =  cv2.imread('.\\pics\\1.jpg',0)
twoImg= cv2.imread('.\\pics\\2.jpg',0)
threeImg =cv2.imread('.\\3.jpg',0)
fourImg =cv2.imread('.\\4.jpg',0)
bombImg =cv2.imread('.\\bomb.jpg',0)


w, h = 10, 10

cap  = PIL.ImageGrab.grab().crop(box)
capArray = np.array(cap)
capArray_gray = cv2.cvtColor(capArray, cv2.COLOR_BGR2GRAY)

#funcs
def nextMove(X,Y):
    x2 = np.array(list(x1), dtype=np.float)
    y2 = np.array(list(y1), dtype=np.float)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x2, y2)

    z = preprocessing.scale(data)
    z = z.reshape(1, z.shape[0])
    prediction = clf.predict(z)
    return prediction



def writeExp(data, df, x, y):
    line = preprocessing.scale(data)
    line = line.reshape(1,line.shape[0])
    ww = pd.DataFrame(z)
    ww.to_csv('.\\data\\question.csv', sep=';', mode = 'a')
    xy = (x,y)
        
    with open('.\\data\\answer.csv', 'a') as answer:
        wr_answer = csv.writer(answer, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        wr_answer.writerow(xy)

def readExp(_path):
    arr = []
    with open(_path) as file:
        reader = csv.reader(file, delimiter=';', quotechar='|')
        for row in reader:
            arr.append(row)
    return(arr)        
###########################
# init.
nanXY  = parse_game.getXY(nanImg)
if len(nanXY)>0:
    Xlist = list(nanXY[:,0])
    Ylist = list(nanXY[:,1])

    X = set(Xlist)
    Y = set(Ylist)
    df = pd.DataFrame(-1, sorted(X), sorted(Y))
    df.index.names = ['X-axis']

    ind = [nanXY[:,0], nanXY[:,1]]
    #data = pd.DataFrame(nanXY, columns = ['X', 'Y'])
    data = pd.DataFrame(np.ones(shape = (nanXY.shape[0], 1) ), index = ind, columns=['label'])
    data['number'] = np.nan

else:
    sys.exit("no Minesweeper game on the screen")
#######


###Draw to test

for i in nanXY:
    pt = tuple(i)
    cv2.rectangle(capArray, pt, (pt[0]+w, pt[1] + h), (0,0,255), 1)
cv2.imwrite('res.png', capArray)

#####

#pyautogui.click(nanXY[0][0], nanXY[0][1])
#time.sleep(0.5)
##

pics = updateXY()
df = updateALL(df)
data = updateData(data)

print(df.transpose())



#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(x,y)
#prediction = clf.predict(z)
#print(prediction)



