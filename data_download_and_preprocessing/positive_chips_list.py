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
    parser.add_argument('--positive_chips_lists_paths', type=str, default = "/hpc/home/csr33/bash_files/positive_chips_lists_dir.npy",
                        help='path to positive chips in complete dataset.')
    parser.add_argument('--blocks', type=str, default = "20",
                        help='path to positive chips in complete dataset.')
    args = parser.parse_args()
    return args

def remove_thumbs(path_to_folder_containing_images):
    """ Remove Thumbs.db file from a given folder
    Args: 
    path_to_folder_containing_images(str): path to folder containing images
    Returns:
    None
    """
    if len(glob(path_to_folder_containing_images + "/*.db", recursive = True)) > 0:
        os.remove(glob(path_to_folder_containing_images + "/*.db", recursive = True)[0])

def list_of_lists_positive_chips(chips_positive_path, blocks):
    positive_chips = os.listdir(chips_positive_path)
    positive_chips_lists = [positive_chips[x:x+blocks] for x in range(0, len(positive_chips), blocks)]
    return(positive_chips_lists)
  
def main(args): 
    remove_thumbs(args.chips_positive_path)  
    np.save(args.positive_chips_lists_paths, list_of_lists_positive_chips(args.chips_positive_path, args.blocks))
if __name__ == "__main__":
    ### Get the arguments 
    args = get_args_parse()
    main(args)



