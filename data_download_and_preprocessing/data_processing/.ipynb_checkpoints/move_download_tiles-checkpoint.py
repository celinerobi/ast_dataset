"""
Correct inconsistent labels
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
    parser.add_argument('--parent_directory', type = str, default = "//oit-nas-fe13dc.oit.duke.edu//data_commons-borsuk//",
                        help = 'path to parent directory; the directory of the storge space.')
    parser.add_argument('--complete_dataset_path', type = str, default = "verified/complete_dataset",
                        help = 'path to the verified complete dataset.')
    parser.add_argument('--tracker_file_path', type = str, default = 'outputs/tile_img_annotation_annotator.npy',
                        help = 'The file path of the numpy array that contains the image tracking.')
    parser.add_argument('--tile_names_tile_urls_complete_array_path', type = str, 
                        default = "image_download_azure/tile_name_tile_url_complete_array.npy",
                        help = 'The file path of the numpy array that contains the tile names and tile urls of the complete arrays.')
    args = parser.parse_args()
    return args

def main(args):
    tile_img_annotation = np.load(args.tracker_file_path)
    tile_names_tile_urls_complete_array = np.load(args.tile_names_tile_urls_complete_array_path)

    #create folder to hold tiles in completed dataset
    path_to_tiles_folder_complete_dataset = os.path.join(args.parent_directory, args.complete_dataset_path,"tiles")
    path_positive_images_complete_dataset = os.path.join(args.parent_directory, args.complete_dataset_path,"chips_positive")
    path_to_verified_sets = os.path.join(args.parent_directory, args.complete_dataset_path,"verified//verified_sets")
    
    tile_names_tile_urls_complete_array = fc.add_formatted_and_standard_tile_names_to_tile_names_time_urls(tile_names_tile_urls_complete_array)
    
    #Move already downloaded tiles to completed dataset
    fc.move_tiles_of_verified_images_to_complete_dataset(tile_img_annotation, 
                                                         path_to_tiles_folder_complete_dataset, 
                                                         path_to_verified_sets)

    # Make a list of the tiles moved to completed dataset
    tiles_downloaded_with_ext_list, tiles_downloaded_without_ext_list = fc.tiles_in_complete_dataset(path_to_tiles_folder_complete_dataset)

    #Download tiles
    downloaded_tiles_info = fc.download_tiles_of_verified_images(path_positive_images_complete_dataset, 
                                                 path_to_tiles_folder_complete_dataset, 
                                                 tiles_downloaded_without_ext_list, 
                                                 tile_names_tile_urls_complete_array)
    
    np.save("downloaded_tiles_info.npy", downloaded_tiles_info)

if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)
