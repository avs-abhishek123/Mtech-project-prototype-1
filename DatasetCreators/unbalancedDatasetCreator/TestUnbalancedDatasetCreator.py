from unbalanceCreator.unbalanceDatasetCreator import *
from unbalanceCreator.unbalanceDatasetCreator import UnbalancedDatasetCreator as udc
import pprint as pp
import json

image_dir = "/mc2/SaiAbhishek/Sprint62/STUDCL-9999-Evaluate improve-Model-performance/balancedDataset/train"
labels_path ="/mc2/SaiAbhishek/Sprint62/STUDCL-9999-Evaluate improve-Model-performance/balancedDataset/balancedDatasetLabels.json"

# These are the arguments of the unbalancedDatasetCreator
# labels_path, image_dir,nA=1000, nB=1000
an_object=udc(labels_path, image_dir,nA=2000, nB=1000)

finalUnbalancedDataset=an_object.unbalancer4clsB()
pp.pprint(finalUnbalancedDataset)

storeUnbalancedDatasetLabels="./unbalancedDataset/unbalancedDatasetLabelsRoughwork.json"

file_pointer=open(storeUnbalancedDatasetLabels,'w')
json.dump(finalUnbalancedDataset, file_pointer, indent = 4)
file_pointer.close()      

# unbalanced creator creates a labels JSON file which has nA & nB number of samples