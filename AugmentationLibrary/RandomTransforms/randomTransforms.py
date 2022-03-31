
import pprint as pp
import numpy as np
import cv2
import os
import json
import random

import PIL
from PIL import Image
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw

from typing import Any, Callable,Dict, List, Optional, Sequence, Tuple, Union
# Typing defines a standard notation for Python function and variable type annotations. 
# The notation can be used for documenting code in a concise, standard format, 
# and it has been designed to also be used by static and runtime type checkers, static analyzers, IDEs and other tools.

import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import albumentations as A

from functools import wraps
# Functools module is for higher-order functions that work on other functions. 
# It provides functions for working with other functions and callable objects to use or extend them without completely rewriting them.




### Random Data Augmentation custom functions


def get_num_channels(image):
    return image.shape[2] if len(image.shape) == 3 else 1
def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.
    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.
    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.
    Returns:
        numpy.ndarray: Transformed image.
    """

    @wraps(process_fn)
    def __process_fn(img):
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn


# #### Random Crop Custom Function


def randomcrop_coords_dict(img,x1,y1,crop_width,crop_height):
    random_crop_output_dict=dict()
    random_crop_output_dict['transform'] = "random_crop"
    random_crop_output_dict['Transformed_image_Np_Array_Format']   = img
    random_crop_output_dict['x1']   = x1
    random_crop_output_dict['y1']   = y1  
    random_crop_output_dict['crop_width']   = crop_width
    random_crop_output_dict['crop_height']   = crop_height
    return random_crop_output_dict

    #['random_crop': x1 : value,y1 : value,widht: value, height: value]
def random_crop(image: np.ndarray):
    #min crop ht=None, max...
    height, width,c = image.shape[:3]

    '''
    print("Height of Original Image",height)
    print("Width of Original Image",width)
    print("Number of channels of Original Image",c)
    print("----------------")
    '''
    
    max_crop_height = height //2
    min_crop_height = height //20
    max_crop_width = width //2
    min_crop_width = width //20

    '''
    print("max_crop_height :",max_crop_height)
    print("min_crop_height :",min_crop_height)
    print("max_crop_width :",max_crop_width)
    print("min_crop_width :",min_crop_width)
    print("----------------")
    '''


    crop_height=random.randint(min_crop_height,max_crop_height)
    crop_width=random.randint(min_crop_width,max_crop_width)

    '''
    print("crop_height :",crop_height)
    print("crop_width :",crop_width)
    print("----------------")
    '''
    
    h_start_max = height-crop_height
    h_start_min = 1

    w_start_max = height-crop_width
    w_start_min = 1
    
    h_start=random.randint(h_start_min,h_start_max)
    w_start=random.randint(w_start_min,w_start_max)

    '''
    print("h_start :",h_start)
    print("w_start :",w_start)
    print("----------------")
    '''



    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = w_start, h_start, w_start+crop_width, h_start+crop_height
    #get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)
    Random_crop_dict = dict(); 


    '''
    print("x1 :",x1)
    print("y1 :",y1)
    print("x2 :",x2)
    print("y2 :",y2)
    print("----------------")

    pixel_size=width*height
    print("Pixel Size :",pixel_size)
    print("----------------")

    print("Y Coordinate 1 :",y1)
    print("Y Coordinate 2 :",y2)
    print("X Coordinate 1 :",x1)
    print("X Coordinate 2 :",x2)
    '''
    Random_crop_dict=dict()
    img = image[y1:y2, x1:x2]
    Random_crop_dict=randomcrop_coords_dict(img,x1, y1, crop_width, crop_height)
    
    '''
    print(img)
    print("----------------")
    print(d)
    '''
    return img,Random_crop_dict

'''
# #### Random Sizing


def random_resizing_coords_dict(img,new_width,new_height):
    random_resizing_output_dict=dict()
    random_resizing_output_dict['transform'] = "random_resize"
    random_resizing_output_dict['Transformed_image_Np_Array_Format']   = img
    random_resizing_output_dict['new_width']   = new_width
    random_resizing_output_dict['new_height']   = new_height  

    return random_resizing_output_dict


    #['random_crop': x1 : value,y1 : value,widht: value, height: value]
def random_resize(img,interpolation=cv2.INTER_LINEAR):
    # 1= increase size
    # 0 = reduce size
    Random_resizing_dict=dict()

    #factor_of_change = keeps aspect ration constant and it can be increased or decreased accordingly
    #factor_of_change should be only integer value

    img_height, img_width = img.shape[:2]

    #limit for image height increase is img height to 10times the image height
    new_height_max =img_height*10
    new_height_min =img_height//10

    #limit for image height increase is img height to 10times the image height
    new_width_max =img_width*10
    new_width_min =img_width//10

    #height and width are being chosen randomly
    new_height =random.randint(new_height_min,new_height_max)
    new_width =random.randint(new_width_min,new_width_max)

    #print(new_width)
    #print(new_height)
    resize_fn = _maybe_process_in_chunks(cv2.resize, dsize=(new_width, new_height), interpolation=interpolation)
    Random_resizing_dict=random_resizing_coords_dict(img,new_width,new_height)
    #print("----------------")
    #print(d)
    return resize_fn(img),Random_resizing_dict
      
''' 
    


# #### Random Scaling


def random_scale_coords_dict(img,fx=None,fy=None):
    random_resizing_output_dict=dict()
    img_height, img_width = img.shape[:2]
    random_resizing_output_dict['transform'] = "random_scale"
    random_resizing_output_dict['Transformed_image_Np_Array_Format']   = img
    random_resizing_output_dict['Scaled_Factor']   = fx  
    random_resizing_output_dict['Scaled_x']   = img_width
    random_resizing_output_dict['Scaled_y']   = img_height  

    return random_resizing_output_dict

def random_scale(img, interpolation=cv2.INTER_LINEAR):

    img_height, img_width = img.shape[:2]
    Random_scale_dict=dict()
    #height and width are being chosen randomly for downscaling
    fx_scale_factor =random.uniform(0.1,10)
    # print("fx_scale_factor",fx_scale_factor)
    fy_scale_factor =fx_scale_factor
    # print("fy_scale_factor",fy_scale_factor)
    rescale_fn = _maybe_process_in_chunks(cv2.resize, dsize=None, fx= fx_scale_factor, fy= fy_scale_factor, interpolation=interpolation)
    new_image=rescale_fn(img)
    Random_scale_dict=random_scale_coords_dict(new_image,fx_scale_factor, fy_scale_factor)
    #print("----------------")
    # print(d)
    return rescale_fn(img),Random_scale_dict
    # resize_fn = _maybe_process_in_chunks(cv2.resize, dsize=(decrease_width, decrease_height), interpolation=interpolation)


# #### Random FLip

def random_flip_dict(img,code):
    random_flip_output_dict=dict()
    random_flip_output_dict['transform'] = "random_flip"
    random_flip_output_dict['Transformed_image_Np_Array_Format']   = img
    #random_crop_output_dict['x1']   = x1
    #random_crop_output_dict['y1']   = y1  
    #random_crop_output_dict['crop_width']   = crop_width
    if code==0:
        random_flip_output_dict['type']   = "Vertically flipped"
    elif code==1:
        random_flip_output_dict['type']   = "Horizontally flipped"
    else:
        random_flip_output_dict['type']   = "Both Horizontally & Vertically flipped"        

    return random_flip_output_dict

'''0 means flipping around the x-axis 
and positive value (for example, 1) means flipping around y-axis. 
Negative value (for example, -1) means flipping around both axes.'''

def random_flip(image):
    randomflip_dict=dict()
    code=random.randint(-1,1)
    transformed_image=cv2.flip(image,int(code))
    randomflip_dict=random_flip_dict(transformed_image,code)
    return transformed_image, randomflip_dict


# #### Random Rotate


def random_rotate_coords_dict(img,angle):
    random_rotate_output_dict=dict()
    random_rotate_output_dict['transform'] = "random_rotate"
    random_rotate_output_dict['Transformed_image_Np_Array_Format']   = img
    #random_crop_output_dict['x1']   = x1
    #random_crop_output_dict['y1']   = y1  
    #random_crop_output_dict['crop_width']   = crop_width
    random_rotate_output_dict['angle']   = angle


    return random_rotate_output_dict

    #['random_crop': x1 : value,y1 : value,widht: value, height: value]
    
def random_rotate(img,interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None):
    
    Random_rotate_dict=dict()
    
    angle=random.randint(-90,90)
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    #print(img)
    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
    )
    rotated_image=warp_fn(img)
    #img = image[y1:y2, x1:x2]
    Random_rotate_dict=random_rotate_coords_dict(img,angle)
    '''print(rotated_image)
    print("----------------")
    print(random_rotate_dict)
    '''
    height, width,c = rotated_image.shape[:3]

    '''
    print("Height of Original Image",height)
    print("Width of Original Image",width)
    print("Number of channels of Original Image",c)
    '''
    return rotated_image,Random_rotate_dict



# #### Random Shift Scale Rotate

def random_shift_scale_rotate_dict(img,angle,scale,dx,dy):
    random_shift_scale_rotate_output_dict=dict()
    random_shift_scale_rotate_output_dict['transform'] = "random_shift_scale_rotate"
    random_shift_scale_rotate_output_dict['Transformed_image_Np_Array_Format']   = img
    random_shift_scale_rotate_output_dict['angle']   = angle
    random_shift_scale_rotate_output_dict['scale']   = scale
    random_shift_scale_rotate_output_dict['dx']   = dx
    random_shift_scale_rotate_output_dict['dy']   = dy

    return random_shift_scale_rotate_output_dict

    return random_rotate_output_dict
def  random_shift_scale_rotate(
    img, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None):
    
    random_shift_scale_rotate_dictionary=dict()
    #visualize(img)
    height, width = img.shape[:2]
    angle = random.randint(-90,90)
    scale=random.uniform(0.1, 10)
    shift_max_height_limit=height/2
    shift_max_width_limit=width/2
    dx = random.uniform(0,shift_max_height_limit)
    dy = random.uniform(0,shift_max_width_limit)
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height

    warp_affine_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
    )
    transformed_img=warp_affine_fn(img)
    random_shift_scale_rotate_dictionary=random_shift_scale_rotate_dict(transformed_img,angle,scale,dx,dy)
    
    
    return transformed_img,random_shift_scale_rotate_dictionary


# #### RandomBrightnessContrast

# #### RandomFog

# #### RandomGamma

# #### RandomRain

# #### RandomShadow

# #### RandomSnow

# #### RandomSunFlare

# #### RandomToneCurve

# #### RandomBrightness

# #### RandomContrast

# #### RandomGridShuffle

# ## Read & Display Images

# #### Read image Custom funtion


def readImage(image_path):
    
    imageInBGR= cv2.imread(image_path)
    imageBGR2RGB=cv2.cvtColor(imageInBGR, cv2.COLOR_BGR2RGB)

    return imageBGR2RGB


# #### Display Image


def visualize(image):
    plt.imshow(image)
    plt.axis("OFF")
    plt.show()


# ### Custom Data Generator


class RandomTransformDataGenerator(Sequence):
    
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, image, num_sample):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param num_sample: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        
        self.image = image
        #self.list_IDs = list_IDs
        #self.labels = labels
        #self.image_path = image_path
        #self.mask_path = mask_path
        #self.to_fit = to_fit
        self.num_sample = num_sample
        #self.dim = dim
        #self.n_channels = n_channels
        #self.n_classes = n_classes
        #self.shuffle = shuffle
        #self.on_epoch_end()
        
    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X
    
    def __len__(self):
            # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        
    def random_shift_scale_rotate (self,image,num_sample_each):
        for i in range(num_sample_each):
            random_shift_scale_rotate_dictionary=dict()
            img,random_shift_scale_rotate_dictionary= random_shift_scale_rotate(image)
            transform_type='random_shift_scale_rotate'
            return img,random_shift_scale_rotate_dictionary  
           
    def random_rotate (self,image,num_sample_each):
        for i in range(num_sample_each):
            random_rotate_dict=dict()
            img,random_rotate_dict=random_rotate(image)
            transform_type='random_rotate'
            return img,random_rotate_dict

    def random_crop (self,image,num_sample_each):
        for i in range(num_sample_each):
            random_crop_dict=dict()
            img,random_crop_dict=random_crop(image)
            transform_type='random_crop'
            return img,random_crop_dict       
    
    def random_scale (self,image,num_sample_each):
        for i in range(num_sample_each):
            random_scale_dict=dict()
            img,random_scale_dict=random_scale(image)
            transform_type='random_scale'
            return img,random_scale_dict  

    '''
    def random_resize (self,image,num_sample_each):
        for i in range(num_sample_each):
            random_resize_dict=dict()
            img,random_resize_dict=random_resize(image)
            transform_type='random_resize'
            return img,random_resize_dict   
    '''
    def random_flip (self,image,num_sample_each):
        for i in range(num_sample_each):
            random_flip_dict=dict()
            img,random_flip_dict=random_flip(image)
            transform_type='random_flip'
            return img,random_flip_dict  
        
    def _load_grayscale_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255
        return img
    
    def random_choose(self,image=None, num_samples=1):
        transforms_dict = {
            # 0 : "no_transform",
            1: "random_crop",
            2: "random_scale",
            3: "random_flip",
            4: "random_rotate",
            5: "random_shift_scale_rotate"
        }
        all_random_transforms_dict=dict()
        randomTransforms_id = transforms_dict.keys()
        num_samples_random_transform_list = []

        for i in range(num_samples):
            num_samples_random_transform_list.append([])
            n= random.randint(2,4)
            for j in range(n):
                ele = random.randint(1, 5)
                num_samples_random_transform_list[i].append(ele)

        randomTransforms_list = transforms_dict.values()

        '''for i in range(num_samples):
            for value in num_samples_random_transform_list[i]:
                # print(transforms_dict(value-1))
            
            
                print(value)
            print("======")'''
        transformed_images_list,all_random_transforms_dict=random_transforms_assigner(image,num_samples_random_transform_list)
        #print(transformed_images_list)
        return transformed_images_list,all_random_transforms_dict
        
    def num_and_transform_assigner(self,image,num):
        d=dict()
        # print(num)
        # print(type(num))
        img=image

        if num==1:
            img,d=random_crop(image)
        elif num==2:
            img,d= random_scale(image)
        elif num==3:
            img,d= random_flip(image)
        elif num==4:
            img,d= random_rotate(image)
        elif num==5:
            img,d= random_shift_scale_rotate(image)  
        return img,d
    
    def random_transforms_assigner(self,image=None, num_samples_random_transform_list=None):
        transforms_dict = {
            # 0 : "no_transform",
            1: "random_crop",
            2: "random_scale",
            3: "random_flip",
            4: "random_rotate",
            5: "random_shift_scale_rotate"
        }
        l=[]
        l=transforms_dict.keys()
        transformed_images=[]
        each_transform_info_list=[]
        all_random_transform_info_list=[]    
        all_random_transforms_dict=dict()
        
        transformed_image_id=1
        for single_transform_list_element in num_samples_random_transform_list:
            transformed_image=image

            all_random_transform_info_list.append([])
            final_transformed_single_image=[]
            for num in single_transform_list_element:
                if num in l:
                    transformed_image,transformation_info_dict=num_and_transform_assigner(transformed_image,num)

                    #print(transformed_image)

                    #print(transformation_info_dict)

                    each_transform_info_list.append(transformation_info_dict)
                    #print(each_transform_info_list)
            final_transformed_single_image.append(transformed_image)
            
            
            all_random_transforms_dict=all_random_transforms_dictionary(transformed_image_id,each_transform_info_list)
            print(all_random_transforms_dict)
            transformed_image_id+=1
            
            transformed_images.append(final_transformed_single_image)
        return transformed_images,all_random_transforms_dict
    
    
def random_choose(image=None, num_samples=1):
    transforms_dict = {
        # 0 : "no_transform",
        1: "random_crop",
        2: "random_scale",
        3: "random_flip",
        4: "random_rotate",
        5: "random_shift_scale_rotate"
    }
    all_random_transforms_dict=dict()
    randomTransforms_id = transforms_dict.keys()
    num_samples_random_transform_list = []

    for i in range(num_samples):
        num_samples_random_transform_list.append([])
        n= random.randint(2,4)
        for j in range(n):
            ele = random.randint(1, 5)
            num_samples_random_transform_list[i].append(ele)

    randomTransforms_list = transforms_dict.values()

    '''for i in range(num_samples):
        for value in num_samples_random_transform_list[i]:
          # print(transforms_dict(value-1))
          
          
          print(value)
        print("======")'''
    transformed_images_list,all_random_transforms_dict=random_transforms_assigner(image,num_samples_random_transform_list)
    #print(transformed_images_list)
    return transformed_images_list,all_random_transforms_dict

def all_random_transforms_dictionary(img_id, each_transform_info_list):
    all_random_transforms_dict=dict()
    all_random_transforms_dict[img_id] = each_transform_info_list
    return all_random_transforms_dict

def num_and_transform_assigner(image,num):
    d=dict()
    # print(num)
    # print(type(num))
    img=image

    if num==1:
        img,d=random_crop(image)
    elif num==2:
        img,d= random_scale(image)
    elif num==3:
        img,d= random_flip(image)
    elif num==4:
        img,d= random_rotate(image)
    elif num==5:
        img,d= random_shift_scale_rotate(image)  
    return img,d
                
def random_transforms_assigner(image=None, num_samples_random_transform_list=None):
    transforms_dict = {
        # 0 : "no_transform",
        1: "random_crop",
        2: "random_scale",
        3: "random_flip",
        4: "random_rotate",
        5: "random_shift_scale_rotate"
    }
    l=[]
    l=transforms_dict.keys()
    transformed_images=[]
    each_transform_info_list=[]
    all_random_transform_info_list=[]    
    all_random_transforms_dict=dict()
    
    transformed_image_id=1
    for single_transform_list_element in num_samples_random_transform_list:
        transformed_image=image

        all_random_transform_info_list.append([])
        final_transformed_single_image=[]
        for num in single_transform_list_element:
            if num in l:
                transformed_image,transformation_info_dict=num_and_transform_assigner(transformed_image,num)

                #print(transformed_image)

                #print(transformation_info_dict)

                each_transform_info_list.append(transformation_info_dict)
                #print(each_transform_info_list)
        final_transformed_single_image.append(transformed_image)
        
        
        all_random_transforms_dict=all_random_transforms_dictionary(transformed_image_id,each_transform_info_list)
        # print(all_random_transforms_dict)
        transformed_image_id+=1
        
        transformed_images.append(final_transformed_single_image)
    return transformed_images,all_random_transforms_dict
def all_random_transforms_dictionary(img_id, each_transform_info_list):
    all_random_transforms_dict=dict()
    all_random_transforms_dict[img_id] = each_transform_info_list
    return all_random_transforms_dict            
