import pyautogui
import cv2

import numpy as np
import pandas as pd
import time
import sys

import train
import parse_game as pg

##
link = 'http://www.freeminesweeper.org//minecore.html'
 
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
        'bombFlagged':[cv2.imread('.\\pics\\bombflagged.jpg',0),[], -10, 'bombFlagged']
}


# init.
X,Y = pg.getBorders(pics)
field = pg.updateField(pics, X, Y)

if len(field)>0:
    Xlist = list(field[:,0])
    Ylist = list(field[:,1])
    X = set(Xlist)
    Y = set(Ylist)
    df = pd.DataFrame(-1, sorted(X), sorted(Y))
    df.index.names = ['X-axis']
    ind = [field[:,0], field[:,1]]
    data = pd.DataFrame(np.ones(shape = (field.shape[0], 1) ), index = ind, columns=['label'])
    data['number'] = np.nan
else:
    sys.exit("no Minesweeper game on the screen")
#######



###Draw to test
#pg.draw('res.png', pics['zero'][1])
#pg.draw('res2.png', pics['nan'][1])
pics = pg.updatePics(pics)
df = pg.updateALLdf(df, pics)
data = pg.updateData(data, pics)

print(data)





