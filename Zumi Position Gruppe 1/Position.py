import os
import rpyc
import numpy as np
import pandas as pd
import time
import math
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
from numba import jit
@jit
def filter_pic(datadiff,pic,newarray):
    for i in range(173,940):
        for j in range(261,1298):
            for k in range(0,3):
                 #if(datadiff[i][j][k]>50):
                if((datadiff[i][j][k]>18) and (datadiff[i][j][k]<240)):
                    for l in range(0,3):
                        newarray[i][j][l] = pic[i][j][l]
    return newarray
@jit
def get_positions(pic,x,y):
    for i in range(173,940):
        for j in range(261,1298):
            for k in range(0,3):
                if(pic[i][j][k]>0):
                    x[i][j] = i
                    y[i][j] = j
    return x,y
@jit
def apply_mask(pic,mask):
    for i in range(173,940):
        for j in range(261,1298):
            if(mask[i][j]==0):
                for l in range(0,3):
                    pic[i][j][l] = 0
    return pic
class Position(object):
    def __init__(self,zuminr):
        self.zuminr = zuminr
        img = cv2.imread("mynew.png")
        self.pic2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    def get_lower_upper_colorvalue(self,Number):
        if ((Number==2)|(Number==5)):
            lower_range = np.array([105, 135, 25])
            upper_range = np.array([121, 255, 255])
        elif (Number==1)|(Number==3):
            lower_range = np.array([0, 88, 70])
            upper_range = np.array([7, 229, 179])
        elif (Number==4)|(Number==6):
            lower_range = np.array([16, 102, 107])
            upper_range = np.array([39, 218, 199])
        return lower_range,upper_range
    def preCalc(self,pic1,datadiff):
        arr = np.zeros(datadiff.shape,np.uint8)
        new_pic = filter_pic(datadiff,pic1,arr)
        hsv = cv2.cvtColor(new_pic, cv2.COLOR_RGB2HSV)
        lower_range,upper_range = self.get_lower_upper_colorvalue(self.zuminr)
        mask = cv2.inRange(hsv, lower_range, upper_range)
        return mask
    def getGlobPos(self,pic1):
         # Umgebung mit Zumis
        img_zwei = pic1
        pic1 = cv2.cvtColor(img_zwei, cv2.COLOR_BGR2RGB)
        datadiff= pic1-self.pic2
        mask = self.preCalc(pic1,datadiff)
        (contours,_) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        points = list()
        for contour in contours:
            area = cv2.contourArea(contour)
            if(area>10):
                print(area)
            if area > 350 and area < 1000:
                (x,y,w,h) = cv2.boundingRect(contour)
                cv2.rectangle(mask, (x,y), (x+w,y+h), (255,255,255), 2)
                points.append((x+0.5*w, y+0.5*h))
        global_pos = np.zeros(2)
        for point in points:
            global_pos[0] += point[0]
            global_pos[1] += point[1]
        global_pos = [int(coord / len(points)) for coord in global_pos]
        return global_pos
    def getAllZumis(self,pic1):
        erg=[]
        zum = self.zuminr
        self.zuminr=1
        erg.append(self.getGlobPos(pic1))
        self.zuminr=2
        erg.append(self.getGlobPos(pic1))
        self.zuminr=4
        erg.append(self.getGlobPos(pic1))
        self.zuminr=zum
        return erg
    def getMoreZumis(self,pic1,zumarr):
        erg=[]
        zum = self.zuminr
        for eintrag in zumarr:
            self.zuminr=eintrag
            erg.append(self.getGlobPos(pic1))
        self.zuminr=zum
        return erg
    def getRelativeCoord(x,y):
        newX=x-400
        newY=(y-400)*(-1)
        return newX,newY
    def getGlobalCoord(x,y):
        newX=x+400
        newY=(y-400)*(-1)
        return newX,newY