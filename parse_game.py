import numpy as np
import pandas as pd
import cv2
import PIL
import itertools
import sys
import csv



dpix = 10
threshold = 0.8
distance = 10
box = (5,100,700,500)
cells = [cv2.imread(".\\pics\\bottomleft.jpg", 0), cv2.imread(".\\pics\\topright.jpg", 0) ] 
    
def getBorders(pics):
    cap = PIL.ImageGrab.grab().crop(box)
    capArray = np.array(cap)
    capArray_gray = cv2.cvtColor(capArray, cv2.COLOR_BGR2GRAY)

    cols  = 'A B C D E F G H I J K L M N O P R S'.split()
    rows = '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16'.split()
    X = []
    Y = []
    for img in cells:
        XYloc = cv2.matchTemplate(capArray_gray, img, cv2.TM_CCOEFF_NORMED)
        XYtupple = np.where(XYloc >= 0.8)
        i0min = 1000
        i1min = 1000
        for i in zip(*XYtupple[::-1]):
            if i[0] < i0min: i0min = i[0]
            if i[1] < i1min: i1min = i[1]           
        if X.count(i0min)==0: X.append(i0min)
        if Y.count(i1min)==0: Y.append(i1min)

    return sorted(X),sorted(Y)

def updateField(pics, X, Y):
    cap = PIL.ImageGrab.grab().crop(box)
    capArray = np.array(cap)
    capArray_gray = cv2.cvtColor(capArray, cv2.COLOR_BGR2GRAY)
    Ylist, Xlist, Xarr, Yarr =  [], [], [], []

    for key in pics:
        XYloc = cv2.matchTemplate(capArray_gray, pics[key][0], cv2.TM_CCOEFF_NORMED)
        XYtupple = np.where(XYloc >= threshold)    
        for xy in zip(*XYtupple[::-1]):
    
            flagY, flagX = 0,0
            if X[0] <= xy[0] <= X[1] and Y[0] <=xy[1] < Y[1]:                 
                for i in range(-5,7): 
                    for j in range(-5,7):
                        if Xarr.count(xy[0]+i)>0 and i!=0:
                            #if (abs(i)+abs(j))!=0:
                            flagX = 1
                        if Yarr.count(xy[1]+j)>0 and j!=0:
                            flagY = 1
                if xy[0] == 494: print(key)
                                    
                
                if flagX==0:    
                    Xarr.append(xy[0])
                if flagY==0:    
                    Yarr.append(xy[1])

    #XYarr = clearDup(XYarr)
    XYarr = np.array(list(itertools.product(Xarr,Yarr)))

    return XYarr


def updatePics(pics):
    cap = PIL.ImageGrab.grab().crop(box)
    capArray = np.array(cap)
    capArray_gray = cv2.cvtColor(capArray, cv2.COLOR_BGR2GRAY)
    for key in pics:
        XYloc = cv2.matchTemplate(capArray_gray, pics[key][0], cv2.TM_CCOEFF_NORMED)
        XYtupple = np.where(XYloc >= threshold)
        pics[key][1] = []    
        for xy in zip(*XYtupple[::-1]):
            pics[key][1].append(xy)
            for key2 in pics:
                for i in range(-5,5): 
                    for j in range(-5,5):
                        if pics[key2][1].count((xy[0] + i, xy[1]+j))>0 and (i!=0 and j!=0) and key!=key2:
                            pics[key][1].remove(xy)
        pics[key][1] = clearDup(pics[key][1])
    return pics



def clearDup(XY):
    "This will clear duplicates from Dataframe"
    for xy in XY:
        for i in range(0, dpix+1):
            for j in range(0, dpix+1):
                if (abs(i) + abs(j)) != 0:
                    if XY.count((xy[0] + i, xy[1] + j)):
                        XY.remove((xy[0] + i, xy[1] + j))
                    if XY.count((xy[0] + j, xy[1] + i)):
                        XY.remove((xy[0] + j, xy[1] + i))      

    return XY




def updateALLdf(df, pics):
    for key in pics:
        if pics[key][2]<10:
            df = updateDF(df, pics[key][1], pics[key][2] )

    return df

def updateData(_data, pics):
    for key in pics:
        if  len(pics[key][1])>0 :
            for xy in pics[key][1]:
                x = xy[0]
                y = xy[1]
                for i in range(-5, 5):
                    for j in range(-5, 5):                     
                        if (x+i, y+j ) in _data.index:
                            _data.loc[(x+i, y+j), 'number'] = pics[key][2]
    return _data

def updateDF(_df: pd.DataFrame, _XY, _val):
    for i0 in _df.axes[0]:
        for i1 in _df.axes[1]:
            for j in _XY:
                if abs(i0-j[0])< dpix and abs(i1-j[1])< dpix :
                    _df.loc[i0].loc[i1] = _val
    return _df

###


def draw(imgName, arr):
    cap = PIL.ImageGrab.grab().crop(box)
    capArray = np.array(cap)
    for i in arr:
        pt = tuple(i)
        cv2.rectangle(capArray, pt, (pt[0]+10, pt[1] + 10), (0,0,255), 1)
    cv2.imwrite(imgName, capArray)
#funcs


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