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
    parser.add_argument('--blocks', type=str, default = "20",
                        help='path to positive chips in complete dataset.')
    parser.add_argument('--block', type=str, default = "20",
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
    positive_chips_lists = [positive_chips[x:x+int(blocks)] for x in range(0, len(positive_chips), int(blocks))]
    return(positive_chips_lists)

def identify_identical_images(images_dir_path, blocks, block):#o_images = None,):
    """
    Args:
    images_dir_path(str): path to directory containing images of interest
    Returns
    same_images(list of lists): lists of images that contain that same information
    https://pysource.com/2018/07/19/check-if-two-images-are-equal-with-opencv-and-python/
    """                         
    same_images_o_images = [] #Make a list to hold the identical images
    same_images_d_images = [] #Make a list to hold the identical images
         
    #Make a list of the images to check for duplicates (images in directory or provided as arugment in function)
    d_images = os.listdir(os.path.join(images_dir_path))

    o_images = list_of_lists_positive_chips(images_dir_path, int(blocks))[int(block)]

    for o in tqdm.tqdm(range(len(o_images))):
        o_image = o_images[o]
        original = cv2.imread(os.path.join(images_dir_path, o_image)) #open image

        for d_image in d_images:
            duplicate = cv2.imread(os.path.join(images_dir_path, d_image)) #open image

            #check for similar characteristics
            if original.shape == duplicate.shape:
                difference = cv2.subtract(original, duplicate)
                b, g, r = cv2.split(difference)

            if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                if o_image != d_image:
                    same_images_o_images.append([o_image]) #Make a list to hold the identical images
                    same_images_d_images.append([d_image]) #Make a list to hold the identical images
                    if d_image in o_images:
                        o_images.remove(d_image) #remove duplicate images, because you have already at least one version to use to find others
        
        d_images.remove(o_image) #remove o_image from d_images list, because you have already checked it against each image
    
    same_images = np.array(same_images_o_images, same_images_d_images)
    return(same_images)
    
def main(args): 
    positive_chips_lists = list_of_lists_positive_chips(args.chips_positive_path,20)
    same_images = identify_identical_images(args.chips_positive_path, args.blocks, args.block)
    np.save("/hpc/group/borsuklab/cred/same_images"+args.block+".npy", same_images)
    
if __name__ == "__main__":
    ### Get the arguments 
    args = get_args_parse()
    main(args)