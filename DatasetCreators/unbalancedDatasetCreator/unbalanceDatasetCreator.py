import os
import pprint as pp
import numpy as np
import cv2
import IPython
import os
import json
import random
import PIL
import urllib
from PIL import Image
from torchvision import transforms
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
import tensorflow as tf
from typing import List, Optional, Sequence, Tuple, Union
import requests
from io import BytesIO
import math
from typing import Any, Callable,Dict, List, Optional, Sequence, Tuple, Union
import glob
import matplotlib.pyplot as plt
import shutil 
import os 
import base64
import torch
import albumentations as A
from functools import wraps


class UnbalancedDatasetCreator():
    def __init__(self, labels_path, image_dir,nA=1000, nB=1000):
        self.labels_path = labels_path
        self.image_dir = image_dir
        self.nA=nA
        self.nB=nB

        json_file = open(self.labels_path)
        self.labelJSON = json.load(json_file)
        json_file.close()

        self.lst2dict4cls

    def lst2dict4cls(self,unbalanced_label_classA,unbalanced_label_classB):
        
        final_unbalancedLabelsDict=dict()
        for itemA in unbalanced_label_classA:
            final_unbalancedLabelsDict[itemA]=1
        for itemB in unbalanced_label_classB:
            final_unbalancedLabelsDict[itemB]=2
        return final_unbalancedLabelsDict             
       

    def unbalancer4clsA(self):
        unbalanced_label_classB=[]
        unbalanced_label_classA=[]
        final_unbalancedLabelsDict=dict()

        for key, value in self.labelJSON.items():
            if value ==2:
                unbalanced_label_classB.append(key)
            else:
                unbalanced_label_classA.append(key)

        finalUnbalancedClassA=random.sample(unbalanced_label_classA, self.nA)
        random.seed(60)

        final_unbalancedLabelsDict=self.lst2dict4cls(finalUnbalancedClassA,unbalanced_label_classB)
        return final_unbalancedLabelsDict

    def unbalancer4clsB(self):
        unbalanced_label_classA=[]
        unbalanced_label_classB=[]
        final_unbalancedLabelsDict=dict()

        for key, value in self.labelJSON.items():
            if value ==1:
                unbalanced_label_classA.append(key)
            else:
                unbalanced_label_classB.append(key)
        
        finalUnbalancedClassB=random.sample(unbalanced_label_classB, self.nB)
        random.seed(60)

        final_unbalancedLabelsDict=self.lst2dict4cls(unbalanced_label_classA,finalUnbalancedClassB)
        return final_unbalancedLabelsDict



