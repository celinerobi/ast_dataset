"""
Functions to process, format, and conduct calculations on the annotated or verified dataset
"""
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

def add_formatted_and_standard_tile_names_to_tile_names_time_urls(tile_names_tile_urls):
    #get a list of the formated tile names
    tile_names = []
    for tile_url in tile_names_tile_urls:
        tile_url = tile_url[1].rsplit("/",3)
        #get the quad standard tile name 
        tile_name = tile_url[3]
        tile_name = os.path.splitext(tile_name)[0] 
        #format tile_names to only include inital capture date 1/20
        if tile_name.count("_") > 5:
            tile_name = tile_name.rsplit("_",1)[0]
        #get the tile name formated (1/14/2022)
        tile_name_formatted = tile_url[1] + "_" + tile_url[2] + "_" + tile_url[3]
        tile_name_formatted = os.path.splitext(tile_name_formatted)[0] 
        tile_names.append([tile_name, tile_name_formatted])
    
    #create array that contains the formated tile_names and tile_names
    tile_names_tile_urls_formatted_tile_names = np.hstack((tile_names_tile_urls, np.array(tile_names)))
    
    return(tile_names_tile_urls_formatted_tile_names)


def unique_formatted_standard_tile_names(tile_names_tile_urls_complete_array):
    unique_tile_name_formatted, indicies = np.unique(tile_names_tile_urls_complete_array[:,3], return_index = True)
    tile_names_tile_urls_complete_array_unique_formatted_tile_names = tile_names_tile_urls_complete_array[indicies,:]
    print("unique formatted tile names", tile_names_tile_urls_complete_array_unique_formatted_tile_names.shape) 

    unique_tile_name_standard, indicies = np.unique(tile_names_tile_urls_complete_array[:,2], return_index = True)
    tile_names_tile_urls_complete_array_unique_standard_tile_names = tile_names_tile_urls_complete_array[indicies,:]
    print("unique standard tile names", tile_names_tile_urls_complete_array_unique_standard_tile_names.shape) 
    
    return(tile_names_tile_urls_complete_array_unique_standard_tile_names, tile_names_tile_urls_complete_array_unique_formatted_tile_names)

def jpg_path_to_tile_name_formatted(tile_paths):
    tile_names = []
    for tile_path in tile_paths:
        base = os.path.basename(tile_path)
        jpg = os.path.splitext(base)[0] #name of tif with the extension removed
        tile_name_formated_name = jpg.rsplit("_",1)[0] #name of tif with the extension removed
        tile_names.append(tile_name_formated_name)
    return(tile_names)

def jpg_path_to_jpg_name_formatted(jpg_paths):
    jpgs_ext = []
    jpgs_without_ext = []
    for jpg_path in jpg_paths:
        jpg_ext = os.path.basename(jpg_path)
        jpg_without_ext = os.path.splitext(jpg_ext)[0] #name of tif with the extension removed
        jpgs_ext.append(jpg_ext)
        jpgs_without_ext.append(jpg_without_ext)
    return(jpgs_ext, jpgs_without_ext)

def remove_thumbs(path_to_folder_containing_thumbs):
    """ Remove Thumbs.db file from a given folder
    Args: 
    path_to_folder_containing_thumbs(str): path to folder containing thumbs.db
    Returns:
    None
    """
    os.remove(glob(path_to_folder_containing_thumbs + "/*.db", recursive = True)[0])
    
def unique_positive_jpgs_from_parent_directory(parent_directory):
    files = []
    paths = []
    counter = 0
    # r=root, d=directories, f = files
    # https://mkyong.com/python/python-how-to-list-all-files-in-a-directory/
    for r, d, f in os.walk(parent_directory):
        folder_name = os.path.basename(r) #identify folder name
        if 'chips_positive' == folder_name: #Specify folders that contain positive chips
            for file in f:
                if '.jpg' in file:
                    paths.append(os.path.join(r, file))
                    files.append(file)
                    counter += 1
    positive_jpgs = np.array((files,paths)).T
    unique_tile_name_formatted_positive_jpgs, indicies = np.unique(positive_jpgs[:,0], return_index = True)
    unique_positive_jpgs = positive_jpgs[indicies]
    print(unique_positive_jpgs.shape)

    return(unique_positive_jpgs)

def image_characteristics(tiles_dir, unique_positive_jpgs):
    """
    Args:
    tiles_dir(str): path to the directory containing tiles
    Returns:
    image_characteristics(pandadataframe):containing image characterisitcs 
    """
    #initialize lists
    state = []
    resolution = []
    year = []
    capture_date  = []
    utm_zone  = []

    standard_tile_name = []
    six_digit_chip_name = []
    NW_coordinates = []
    SE_coordinates = []
    row_indicies = []
    col_indicies = []

    for tile_name in tqdm_notebook(os.listdir(tiles_dir)): #index over the tiles in the tiles_dir 
        file_name, ext = os.path.splitext(tile_name) # File name
        count = 1      
        
        item_dim = int(512)          
        tile = cv2.imread(os.path.join(tiles_dir, tile_name)) 
        tile_height,  tile_width,  tile_channels = tile.shape #the size of the tile 

        #divide the tile into 512 by 512 chips (rounding up)
        row_index = math.ceil(tile_height/512) 
        col_index = math.ceil(tile_width/512)

        for x in range(0, col_index):
            for y in range(0, row_index):
                six_digit_chip_name_temp = file_name+ '_'+ str(count).zfill(6) + '.jpg'
                count += 1  
                if six_digit_chip_name_temp in unique_positive_jpgs[:,0]: #only record values for images that are annotated
                    #image characteristics
                    six_digit_chip_name.append(six_digit_chip_name_temp) # The index is a six-digit number like '000023'.
                    NW_coordinates.append([x*item_dim, y*(item_dim)]) #NW (Top Left) 
                    SE_coordinates.append([x*item_dim+item_dim-1, y*(item_dim)+item_dim-1]) #SE (Bottom right) 
                    row_indicies.append(y)
                    col_indicies.append(x)
                    #tile characteristics
                    standard_tile_name.append(file_name.split("_",4)[4]) #standard_tile_name
                    state.append(file_name.split("_",9)[0]) #state
                    resolution.append(file_name.split("_",9)[1]) #resolution
                    utm_zone.append(file_name.split("_",9)[7]) #utm
                    year.append(file_name.split("_",9)[2]) #year
                    capture_date.append(file_name.split("_",9)[-1]) #capture date
                    
    #create pandas dataframe
    image_characteristics = pd.DataFrame(data={'state': state,
                                                'resolution': resolution,
                                                'year': year,
                                                'capture_date':capture_date,
                                                'utm_zone': utm_zone,
                                                    'standard_tile_name': standard_tile_name,
                                                'six_digit_chip_name':six_digit_chip_name,
                                                'NW_pixel_coordinates': NW_coordinates,
                                                'SE_pixel_coordinates': SE_coordinates,
                                                'row_indicies': row_indicies,
                                                'col_indicies': col_indicies})
    return(image_characteristics)