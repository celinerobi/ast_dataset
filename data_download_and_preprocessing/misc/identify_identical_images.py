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
#import data_eng.form_calcs as fc
import cv2
import tqdm
import argparse
from glob import glob


def get_args_parse():
    parser = argparse.ArgumentParser(
        description='This script records the annotator who has labeled each image')
    parser.add_argument('--chips_positive_path', type=str, default = "C:/chip_allocation/complete_dataset/chips_positive",
                        help='path to positive chips in complete dataset.')
    parser.add_argument("-B",'--blocks', type=str, default = "20",
                        help='path to positive chips in complete dataset.')
    parser.add_argument("-b",'--block', type=str, default = "20",
                        help='path to positive chips in complete dataset.')
    args = parser.parse_args()
    return args


    
def main(args): 
    chips_positive_path = "C:/chip_allocation/complete_dataset/chips_positive"
    unique_chips_positive_path = "C:/chip_allocation/complete_dataset/unique_chips_positive"
    dups_chips_positive_path = "C:/chip_allocation/complete_dataset/dups_chips_positive"

    images, imgsr, imgsg, imgsb = fc.positive_images_to_array(chips_positive_path)
    unique_imgsr, duplicate_imgsr =fc.unique_by_first_dimension(imgsr, images)
    unique_imgsg, duplicate_imgsg = fc.unique_by_first_dimension(imgsg, images)
    unique_imgsb, duplicate_imgsb = fc.unique_by_first_dimension(imgsb, images)
    unique_images = fc.intersection_of_sets(unique_imgsr, unique_imgsg, unique_imgsb)
    duplicate_images = fc.intersection_of_sets(duplicate_imgsr, duplicate_imgsg, duplicate_imgsb)   

    fc.move_images(chips_positive_path, unique_chips_positive_path, unique_images)
    fc.move_images(chips_positive_path, dups_chips_positive_path, duplicate_images)

    #positive_chips_lists = list_of_lists_positive_chips(args.chips_positive_path,20)
    #same_images = identify_identical_images(args.chips_positive_path, args.blocks, args.block)
    #np.save("/hpc/group/borsuklab/cred/same_images"+args.block+".npy", same_images)
    
if __name__ == "__main__":
    ### Get the arguments 
    args = get_args_parse()
    main(args)