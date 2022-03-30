from randomTransforms import *
import unittest
import random
import numpy as np

"""
an_object= RandomTransformDataGenerator(image,2)

img_dict=dict()
img,img_dict=an_object.random_choose(image,2)
print(img)
print(img_dict)"""

class Testrandom_choose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        image_path="Random Transforms/dog.jpg"
        image= readImage(image_path)
        
        num_samples=random.randint(1,10)
        
        cls.image=image
        cls.num_samples=num_samples
        
        cls.an_object= RandomTransformDataGenerator(image, num_samples)


    @classmethod    
    def tearDownClass(cls):
        pass
    
    """def test_random_choose(self):
        result_image,result_dict= random_choose(self.image,2)
        self.assertEqual(type(result_image), "list")
        self.assertEqual(type(result_dict), "dict")"""
        
    def test_random_rotate(self):
        dict_123=dict()

        result_image,result_dict= random_rotate(self.image)
        self.assertEqual(type(result_image), np.ndarray)
        self.assertEqual(type(result_dict), type(dict_123))

        
    def test_random_scale(self):
        dict_123=dict()

        result_image,result_dict= random_scale(self.image)
        self.assertEqual(type(result_image), np.ndarray)
        self.assertEqual(type(result_dict), type(dict_123))

        
    def test_random_flip(self):
        dict_123=dict()

        result_image,result_dict= random_flip(self.image)
        self.assertEqual(type(result_image), np.ndarray)
        self.assertEqual(type(result_dict), type(dict_123))

        
    def test_random_shift_scale_rotate(self):
        dict_123=dict()

        result_image,result_dict= random_shift_scale_rotate(self.image)
        self.assertEqual(type(result_image), np.ndarray)
        self.assertEqual(type(result_dict), type(dict_123))

        
    def test_random_crop(self):
        dict_123=dict()

        result_image,result_dict= random_crop(self.image)
        self.assertEqual(type(result_image), np.ndarray)
        self.assertEqual(type(result_dict), type(dict_123))

    #========================================================
        
    def test_random_rotate_inside_class(self):
        dict_123=dict()

        result_image,result_dict= self.an_object.random_rotate(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)
        self.assertEqual(type(result_dict), type(dict_123))

    def test_random_flip_inside_class(self):
        dict_123=dict()

        result_image,result_dict= self.an_object.random_flip(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)
        self.assertEqual(type(result_dict), type(dict_123))    

    def test_random_scale_inside_class(self):
        dict_123=dict()

        result_image,result_dict= self.an_object.random_scale(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)
        self.assertEqual(type(result_dict), type(dict_123))


    def test_random_crop_inside_class(self):
        dict_123=dict()

        result_image,result_dict= self.an_object.random_crop(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)
        self.assertEqual(type(result_dict), type(dict_123))


    def test_random_shift_scale_rotate_inside_class(self):
        dict_123=dict()

        result_image,result_dict= self.an_object.random_shift_scale_rotate(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)
        self.assertEqual(type(result_dict), type(dict_123))

    
    def test_random_choose_inside_class(self):
        dict_123=dict()
        list_123=[]
        result_image,result_dict= self.an_object.random_choose(self.image,self.num_samples)
        self.assertEqual(type(result_image), type(list_123))
        self.assertEqual(type(result_dict), type(dict_123))

    
    def test_num_and_transform_assigner_inside_class(self):
        dict_123=dict()
        list_123=[]
        result_image,result_dict= self.an_object.num_and_transform_assigner(self.image,self.num_samples)
        self.assertEqual(type(result_image), np.ndarray)
        self.assertEqual(type(result_dict), type(dict_123))

    


    




        
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


