import numpy as np
from PIL import Image
from PIL import ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2
import glob
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd

def ListImages(file):    
    """
    imagelist :: list of full paths of images
    imagelist_wo :: list of image filenames
    """
    imagelist = []
    imagelist_wo = []
    folder = file
    for filepath in glob.iglob(folder+'*.jpg'):
        imagelist.append(filepath)        
        imagelist_wo.append(filepath[len(folder):]) 
    return imagelist, imagelist_wo


def CreateDesignVector(imagelist,X=[],n=256):   
    """
    n :: number of pixels per side
    """
    for filepath in imagelist:
        img = Image.open(filepath).convert('RGB')
        arr = np.array(img)
        arr = cv2.resize(arr, (n, n))
        arr = arr.ravel()
        X.append(arr)
    return X


def label_class_to_Integer(Y):
    """
    Y: array 
    num_classes: integer
    return: array Y and number of classes 
    """
    dictY = dict(zip(np.unique(Y),range(len(np.unique(Y)))))
    num_classes = len(dictY)
    for i in range(len(Y)):
        Y[i] = dictY[Y[i]]

    return Y.astype(int), num_classes



def sort_by_frequency_and_relabel(Y, num_classes):
    """
    return:
    list_mapping: list of [old index, new index]
    Y array: new Y after relabeling to new index 
    """
    
    listCount = [0]*num_classes
    for i in range(len(Y)):
        listCount[Y[i]] +=1

    list_mapping = []
    for i in range (num_classes):
        list_mapping.append([listCount[i], i])
    list_mapping.sort()

    #save original mapping 
    lib = list_mapping

    for i in range ((num_classes)):
        new_index = num_classes -i -1
        old_index = list_mapping[i][1]
        
        list_mapping[i][0] = old_index
        list_mapping[i][1] = new_index

    #now sort again based on frequency
    list_mapping.sort()
    for i in range(len(Y)):
        Y[i] = list_mapping[Y[i]][1]

    #return mapping index and new Y
    return list_mapping, Y
