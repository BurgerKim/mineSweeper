import numpy as np
import pandas as pd
import cv2
import PIL

dpix = 10
threshold = 0.8

def getXY(_img):
    cap = PIL.ImageGrab.grab().crop(box)
    capArray = np.array(cap)
    capArray_gray = cv2.cvtColor(capArray, cv2.COLOR_BGR2GRAY)
###
    XYloc = cv2.matchTemplate(capArray_gray, _img, cv2.TM_CCOEFF_NORMED)
    XYtupple = np.where(XYloc >= threshold)
    XY = [(0,0)]
    for i in zip(*XYtupple[::-1]):
        prev = i
        XY.append(i)
    XY.remove((0,0))
    ### 
    prev = (0,0)
    XY = clearDup(XY)
    
    XYarr = np.array(XY)
    return XYarr

def clearDup(XY):
    "This will clear duplicates from Dataframe"
    for xy in XY:
        for i in range(0, dpix+1):
            for j in range(1, dpix+1):
                if XY.count((xy[0] + i, xy[1] + j)):
                    XY.remove((xy[0] + i, xy[1] + j))
                if XY.count((xy[0] + j, xy[1] + i)):
                    XY.remove((xy[0] + j, xy[1] + i))      

    return XY

def updateXY():
    for key in pics:
        pics[key][1] =  getXY(pics[key][0])
    return pics


def updateALL(df):
    for key in pics:
        if pics[key][2]<10:
            df = updateDF(df, pics[key][1], pics[key][2] )

    return df

def updateData(_data):
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
