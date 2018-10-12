#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 09:05:25 2018

@author: mathieuvdc
"""

import random
import csv
import cv2
import numpy as np
import pandas as pd
from matplotlib import *
from skimage import morphology, filters, measure, segmentation, color, io, data
# %% Lecture

imgt1 = pyplot.imread('IRMcoupe17-t1.jpg')
imgt2 = pyplot.imread('IRMcoupe17-t2.jpg')
# %%  affichage image 1
pyplot.imshow(imgt1, cmap='gray')
# %% affichage image 2
pyplot.imshow(imgt2, cmap='gray')

# %% METHOD 1 : BINARIZATION & LABELLING

# %% Comparaison d'histogrammes dans un premier temps

def histogramme(image):
    h = np.zeros(256,dtype=np.uint32)
    s = image.shape
    for j in range(s[0]):
        for i in range(s[1]):
            valeur = image[j,i]
            h[valeur] += 1
    return h
# %% Histogramme 1  
h = histogramme(imgt1)
pyplot.figure(figsize=(8,6))
pyplot.plot(h)
pyplot.axis([0,255,0,10000])
pyplot.xlabel("valeur")
pyplot.ylabel("Nombre")

# %% Histogramme 2  
h2 = histogramme(imgt2)
pyplot.figure(figsize=(8,6))
pyplot.plot(h2)
pyplot.axis([0,255,0,10000])
pyplot.xlabel("valeur")
pyplot.ylabel("Nombre")

# il semble y avoir + de blancs (pic à 9000 en t2 contre pic à 7600 en t1) donc peut-être une tumeur plus grande. 
# %% Binarization
def binarize(image, seuil):
    n,p = image.shape
    result = np.zeros([n,p])
    for i in range(n):
        for j in range(p):
            if image[i,j] >= seuil:
                result[i,j] = 256
            else :
                result[i,j] = 0
    return result
# %% Binarization image 1
# i choose 70, it seems to be a good value.
imgt1b = binarize(imgt1,70)
pyplot.imshow(imgt1b)

# %% Binarization image 2
imgt2b = binarize(imgt2,70)
pyplot.imshow(imgt2b)

# %% Labelling
from scipy.ndimage.measurements import label
# %% Image T1 :
label_t1, nb_feature = label(imgt1b)
# %%
nb_feature
# %%
pyplot.imshow(label_t1) # pour montrer le résultat de la labellisation
# %%
count = np.zeros((nb_feature+1, ))
for i in range(nb_feature + 1):
    pos = label_t1 == i
    count[i] = np.sum(pos.flatten())
tumeurlabel = np.argmax(count[1:])+1
# %%
pos = label_t1 == tumeurlabel
t1_tumor = np.zeros((256, 256))
t1_tumor[pos] = 256
pyplot.imshow(t1_tumor)

# %% Image T2 :
label_t2, nb_feature = label(imgt2b)
# %%
nb_feature
# %%
pyplot.imshow(label_t2) # pour montrer le résultat de la labellisation
# %%
count2 = np.zeros((nb_feature+1, ))
for i in range(nb_feature + 1):
    pos = label_t2 == i
    count2[i] = np.sum(pos.flatten())
tumeurlabel2 = np.argmax(count2[1:])+1
# %%
pos = label_t2 == tumeurlabel2
t2_tumor = np.zeros((256, 256))
t2_tumor[pos] = 256
pyplot.imshow(t2_tumor)

# %% Affichage des tumeurs superposées aux images d'origine
# %% image 1
pyplot.imshow(imgt1,cmap="Greys_r")
pyplot.contour(t1_tumor, origin="lower")
pyplot.title("Image T1")
pyplot.figure()
pyplot.imshow(imgt2,cmap="Greys_r")
pyplot.contour(t2_tumor, origin="lower")
pyplot.title("Image T2")

# %% Mesures
# en nombre de pixels :
tumort1_nb_pixels = count[tumeurlabel] #avec seuil = 70, 1857 pix
print(tumort1_nb_pixels)
tumort2_nb_pixels = count2[tumeurlabel2] #avec seuil = 70, 1871 pix
print(tumort2_nb_pixels)
evolution = (tumort2_nb_pixels-tumort1_nb_pixels)/tumort1_nb_pixels*100
print(evolution) # il y a 0,75 % de grossissement


