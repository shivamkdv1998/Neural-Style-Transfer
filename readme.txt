# Neural Style Transfer with tf.keras

## Problem Statement

The aim of this assignment is to create a deep learning model capable of adapting an existing work to resemble the aesthetic of any art. The model should be able to analyze the artistic style of the selected art and apply similar stylistic features to a new, original artwork, creating a piece that seems as though it could have been created by the artist themselves.

## Approach

Our work is based on [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) by Leon Gatys, Alexander Ecker, and Matthias Bethge.

We used a VGG19 model, pretrained on the Imagenet dataset, for transferring the style of one image to another.

## Setup

The JPG images can be downloaded from internet by providing URL or can be uploaded from the system.
Our defined function works on images of size 512x512. We have defined a function to resize the images to 512x512. 

## Import and configure:
The following libraries are to be imported :
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

import numpy as np
from PIL import Image
import time
import functools
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import image_dataset_from_directory as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
import IPython.display
import os

## Required function are created to visualize and process images

## Following layers are pulled out for style transfer, from feature maps:
content_layers = ['block5_conv2']

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


#functions for finding content loss and style loss are defined


#function run_style_transfer() defined for transfering style of style image to content image

## the above function is called for each image in the content folder, by the function style_transfer_folder(content_folder_path, style_image, output_path)

## show_img() displays the result of run_style_transfer