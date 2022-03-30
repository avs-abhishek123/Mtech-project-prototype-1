from AugmentationLibrary.StaticTransforms.staticTransforms import StaticTransformDataGenerator as AugStatic
import PIL
import cv2
from PIL import Image

augmentationtype=input()
img=cv2.imread("AugmentationLibrary/StaticTransforms/dog.jpg")
image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

an_object= AugStatic(image,1)

# return number of sample, store them in the folder

print("AugStatic."+augmentationtype+"()")


augmented_image=eval("an_object."+augmentationtype+"()")


Image.fromarray(augmented_image).save("/mc2/SaiAbhishek/Sprint62/STUDCL-9999-Evaluate improve-Model-performance/AugmentationLibrary/StaticTransforms/OutputImages/"+augmentationtype+".jpg")

