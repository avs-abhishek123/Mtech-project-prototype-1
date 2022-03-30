from staticTransforms import *
import unittest
import random
import numpy as np

"""
an_object= RandomTransformDataGenerator(image,2)

img_dict=dict()
img,img_dict=an_object.random_choose(image,2)
print(img)
print(img_dict)"""

class TeststaticTransforms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        image_path="Static Transforms/dog.jpg"
        image= readImage(image_path)
        num_samples=1
        #num_samples=random.randint(1,10)
        
        cls.image=image
        cls.num_samples=num_samples
        cls.image_path=image_path
        cls.an_object= StaticTransformDataGenerator(image, num_samples)


    @classmethod    
    def tearDownClass(cls):
        pass
    
    """def test_random_choose(self):
        result_image,result_dict= random_choose(self.image,2)
        self.assertEqual(type(result_image), "list")
        self.assertEqual(type(result_dict), "dict")"""
        
    def test_Blur(self):
        result_image= Blur(self.image)
        self.assertEqual(type(result_image), np.ndarray)


    def test_CLAHE(self):
        result_image= CLAHE(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_ChannelDropout(self):
        result_image= ChannelDropout(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_ChannelShuffle(self):
        result_image= ChannelShuffle(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_ColorJitter(self):
        result_image= ColorJitter(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Downscale(self):
        result_image= Downscale(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Emboss(self):
        result_image= Emboss(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_FancyPCA(self):
        result_image= FancyPCA(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_FromFloat(self):
        result_image= FromFloat(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_GaussNoise(self):
        result_image= GaussNoise(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_GaussianBlur(self):
        result_image= GaussianBlur(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_GlassBlur(self):
        result_image= GlassBlur(self.image)
        self.assertEqual(type(result_image), np.ndarray)


        
    def test_HueSaturationValue(self):
        result_image= HueSaturationValue(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_ISONoise(self):
        result_image= ISONoise(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_InvertImg(self):
        result_image= InvertImg(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_MedianBlur(self):
        result_image= MedianBlur(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_MotionBlur(self):
        result_image= MotionBlur(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_MultiplicativeNoise(self):
        result_image= MultiplicativeNoise(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Normalize(self):
        result_image= Normalize(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Posterize(self):
        result_image= Posterize(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_RGBShift(self):
        result_image= RGBShift(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Sharpen(self):
        result_image= Sharpen(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Solarize(self):
        result_image= Solarize(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Superpixels(self):
        result_image= Superpixels(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_ToFloat(self):
        result_image= ToFloat(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_ToGray(self):
        result_image= ToGray(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_ToSepia(self):
        result_image= ToSepia(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_VerticalFlip(self):
        result_image= VerticalFlip(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_HorizontalFlip(self):
        result_image= HorizontalFlip(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Flip(self):
        result_image= Flip(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Transpose(self):
        result_image= Transpose(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_OpticalDistortion(self):
        result_image= OpticalDistortion(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_GridDistortion(self):
        result_image= GridDistortion(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_JpegCompression(self):
        result_image= JpegCompression(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Cutout(self):
        result_image= Cutout(self.image)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_CoarseDropout(self):
        result_image= CoarseDropout(self.image)
        self.assertEqual(type(result_image), np.ndarray)


        
    def test_GridDropout(self):
        result_image= GridDropout(self.image)
        self.assertEqual(type(result_image), np.ndarray)


    def test_readImage(self):
        result_image= readImage(self.image_path)
        self.assertEqual(type(result_image), np.ndarray)

 
    #========================================================
       
    def test_Blur_inside_class(self):
        result_image = self.an_object.Blur(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_CLAHE_inside_class(self):
        result_image = self.an_object.CLAHE(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_ChannelDropout_inside_class(self):
        result_image = self.an_object.ChannelDropout(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_ChannelShuffle_inside_class(self):
        result_image = self.an_object.ChannelShuffle(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_ColorJitter_inside_class(self):
        result_image = self.an_object.ColorJitter(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Downscale_inside_class(self):
        result_image = self.an_object.Downscale(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Emboss_inside_class(self):
        result_image = self.an_object.Emboss(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_FancyPCA_inside_class(self):
        result_image = self.an_object.FancyPCA(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_FromFloat_inside_class(self):
        result_image = self.an_object.FromFloat(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_GaussNoise_inside_class(self):
        result_image = self.an_object.GaussNoise(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_GaussianBlur_inside_class(self):
        result_image = self.an_object.GaussianBlur(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_GlassBlur_inside_class(self):
        result_image = self.an_object.GlassBlur(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_HueSaturationValue_inside_class(self):
        result_image = self.an_object.HueSaturationValue(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_ISONoise_inside_class(self):
        result_image = self.an_object.ISONoise(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_InvertImg_inside_class(self):
        result_image = self.an_object.InvertImg(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_MedianBlur_inside_class(self):
        result_image = self.an_object.MedianBlur(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_MotionBlur_inside_class(self):
        result_image = self.an_object.MotionBlur(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_MultiplicativeNoise_inside_class(self):
        result_image = self.an_object.MultiplicativeNoise(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Normalize_inside_class(self):
        result_image = self.an_object.Normalize(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Posterize_inside_class(self):
        result_image = self.an_object.Posterize(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_RGBShift_inside_class(self):
        result_image = self.an_object.RGBShift(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Sharpen_inside_class(self):
        result_image = self.an_object.Sharpen(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Solarize_inside_class(self):
        result_image = self.an_object.Solarize(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Superpixels_inside_class(self):
        result_image = self.an_object.Superpixels(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_ToFloat_inside_class(self):
        result_image = self.an_object.ToFloat(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_ToGray_inside_class(self):
        result_image = self.an_object.ToGray(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_ToSepia_inside_class(self):
        result_image = self.an_object.ToSepia(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_VerticalFlip_inside_class(self):
        result_image = self.an_object.VerticalFlip(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_HorizontalFlip_inside_class(self):
        result_image = self.an_object.HorizontalFlip(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Flip_inside_class(self):
        result_image = self.an_object.Flip(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Transpose_inside_class(self):
        result_image = self.an_object.Transpose(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_OpticalDistortion_inside_class(self):
        result_image = self.an_object.OpticalDistortion(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_GridDistortion_inside_class(self):
        result_image = self.an_object.GridDistortion(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_JpegCompression_inside_class(self):
        result_image = self.an_object.JpegCompression(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_Cutout_inside_class(self):
        result_image = self.an_object.Cutout(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_CoarseDropout_inside_class(self):
        result_image = self.an_object.CoarseDropout(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)

        
    def test_GridDropout_inside_class(self):
        result_image = self.an_object.GridDropout(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)



        
'''        
class Testrandom_rotate(unittest.TestCase):
    
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
'''
if __name__ == '__main__':
    unittest.main()


