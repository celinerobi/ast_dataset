"""
Record the tile, chip, and annotation file names, chip pathway, and annotator name in a csv and npy array after each worker has reviewed their annotations.
"""

"""
Import Packages
"""
# Standard packages
#import warnings
#import urllib
#import shutil
import os
# Less standard, but still pip- or conda-installable
import numpy as np

#import rasterio
#import re
#import rtree
#import shapely
#import pickle
#import data_eng.az_proc as ap
import data_eng.form_calcs as fc
import cv2
import tqdm
import argparse
from glob import glob


def get_args_parse():
    parser = argparse.ArgumentParser(
        description='This script records the annotator who has labeled each image')
    parser.add_argument('--chips_positive_path', type=str, default = "C:/chip_allocation/complete_dataset/chips_positive",
                        help='path to positive chips in complete dataset.')
    args = parser.parse_args()
    return args
    
def main(args): 
    fc.remove_thumbs(args.chips_positive_path)  
    positive_chips_lists = fc.list_of_lists_positive_chips(args.chips_positive_path)
    same_images = fc.identify_identical_images(args.chips_positive_path, args.chips_positive_path, positive_chips_lists[0])
    np.save("same_images.npy", same_images)
    
if __name__ == "__main__":
    ### Get the arguments 
    args = get_args_parse()
    main(args)



