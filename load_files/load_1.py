# Provides services to interface with an underlying operating system
import os
# Provides functions for duplicating objects using shallow or deep copy semantics.
import copy
# aggreation of unix style path name pattern processing services
import glob
# aggretation of services required for multithreading system
import threading
# aggregation of efficient array operations
import numpy as np
# aggregation of scientific and numerical tools
import scipy.io
import scipy.ndimage as spnd
# aggregation of efficient tensor operations
import tensorflow as tf
# PIL = Python Imaging Library
from PIL import Image


# Global Variables
current_working_directory = os.getcwd()
# Datapath for AFLW code
data_path = "/home/ekram67/Datasets/AFLW"
version_number = ""

# Classes
class AFLW_Dataset:
	# Constructors
	def __init__(self,idx,data_partition):
		# Creating instance variables
    		self.data_path = "/home/ekram67/Datasets"
       		self.version_number = ""
        	# Determining file path of dataset
        	self.dir = os.path.join(data_path,version_number)
        	self.dataset = "AFLW"
		self.name = '{}_{}'.format(self.dataset,self.version_number)
		# get list of all files - use build in open() function rather the PIL routine
		self.all_idxs = open('{}/{}'.format(self.data_path,data_partition)).read().splitlines()
	def num_imgs(self):
		return len(self.all_idxs)
	def load_image(self, idx):
		im = Image.open(os.path.join(self.data_path,self.dataset,'/{}.jpg'.format(idx)))
		im = np.array(im, dtype=np.float32)
		return im

# Main

dataset = AFLW_Dataset('1000-image66034','training.txt')

