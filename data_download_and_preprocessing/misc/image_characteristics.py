"""
Create pandas csv of image characteristics  
"""

"""
Import Packages
"""
import argparse
# Standard packages
import tempfile
import warnings
import urllib
import shutil
import os
# Less standard, but still pip- or conda-installable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import rasterio
import re
import rtree
import shapely
import pickle
import data_eng.az_proc as ap
import data_eng.form_calcs as fc

#from cartopy import crs
import collections
import cv2
import math
from glob import glob
from tqdm.notebook import tqdm_notebook

def get_args_parse():
    parser = argparse.ArgumentParser(
        description='This script adds a subdirectory of xmls to correct possible inconsistent labels')
    parser.add_argument('--parent_dir', type = str, default = "//oit-nas-fe13dc.oit.duke.edu//data_commons-borsuk//",
                        help = 'path to parent directory; the directory of the storge space.')
    parser.add_argument('--complete_dataset_path', type = str, default = "verified/complete_dataset",
                        help = 'path to the verified complete dataset.')
    args = parser.parse_args()
    return args

def main(args):
    #specify folder that holds tiles in completed dataset
    tiles_complete_dataset_path = os.path.join(args.parent_dir, args.complete_dataset_path, "tiles")

    #unique positive jpgs (file names with the file extension)
    unique_positive_jpgs = fc.unique_positive_jpgs_from_parent_directory(args.parent_dir)
    
    image_characteristics = fc.image_characteristics(tiles_complete_dataset_path, unique_positive_jpgs)

    image_characteristics.to_csv('image_characteristics.csv')
    counterin = 0
    counternot = 0 

    #Check to see how many images are not yet in the image characteristics folder (not verified)
    for unique_jpg in unique_positive_jpgs[:,0]:
        if image_characteristics['six_digit_chip_name'].isin([unique_jpg]).any():
            counterin += 1
        if not image_characteristics['six_digit_chip_name'].isin([unique_jpg]).any():
            counternot += 1
    print("images included in the image characteristics csv ",counterin, \
          "images not included in the image characteristics csv \ left to be verified", counternot)

if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)
